import json
import os
import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from scipy.stats import rankdata
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from predwordimp.eval.metrics import RankingEvaluator
from predwordimp.eval.util import get_model_name, get_rank_limit
from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

logger = get_logger(__name__)


def load_ds() -> Dataset:
    return load_dataset("adasgaleus/word-importance", split="test")  # type: ignore


def is_prohibited_word(w: str) -> bool:
    if w.startswith("(PERSON"):  # speaker tags
        return True
    return False


@dataclass
class EvalWordImp(ConfigurableJob):
    hf_model: str
    seed: int = 69
    stride: int = 128
    max_rank_limit: int | float = 0.1

    job_version: str = "0.4"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModelForTokenClassification.from_pretrained(self.hf_model)

    def tokenize(self, ds: Dataset) -> BatchEncoding:
        tokenized = self.tokenizer(
            ds["context"],
            truncation=True,
            padding=True,
            # return_overflowing_tokens=True,
            # return_special_tokens_mask=True,
            stride=self.stride,
            is_split_into_words=True,
            return_tensors="pt",
        )

        self.set_prohibited(tokenized, ds)

        return tokenized

    def set_prohibited(self, tokenized: BatchEncoding, ds: Dataset) -> None:
        self.prohibited_word_ids = torch.zeros(tokenized.data["input_ids"].shape)
        for batch_idx, context in enumerate(ds["context"]):
            for word_idx, word in enumerate(context):
                if is_prohibited_word(word):
                    self.prohibited_word_ids[batch_idx, word_idx] = True

        logger.debug(f"prohibited_word_ids: {self.prohibited_word_ids}")

    def predict(self, tokenized_inputs: BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(**tokenized_inputs).logits
        return logits

    @staticmethod
    def get_ignore_mask(
        tokenized_inputs: BatchEncoding, prohibited: torch.Tensor
    ) -> torch.Tensor:
        subword_masks = []

        for batch_idx in range(tokenized_inputs.data["input_ids"].shape[0]):
            word_ids = tokenized_inputs.word_ids(batch_idx)
            previous_word_idx = None
            subword_mask = []
            for word_idx in word_ids:
                if word_idx is None:
                    subword_mask.append(0)
                elif prohibited[batch_idx, word_idx]:
                    subword_mask.append(1)
                elif word_idx != previous_word_idx:
                    subword_mask.append(0)
                else:
                    subword_mask.append(1)
                previous_word_idx = word_idx
            subword_masks.append(subword_mask)

        return torch.tensor(subword_masks)

    @staticmethod
    def sorted2ordering(
        sorted_indices: torch.Tensor,
        tokenized_inputs: BatchEncoding,
        rank_limit: float | int,
    ) -> list[tuple[list[int], int]]:
        rankings = []

        for batch_idx, pred in enumerate(sorted_indices):
            ranking = []
            word_ids = tokenized_inputs.word_ids(batch_idx)
            num_words = max(wi for wi in word_ids if wi is not None) + 1
            limit = get_rank_limit(rank_limit, num_words)

            for index in pred:
                word_id = word_ids[index]

                if word_id is None:
                    continue

                ranking.append(word_id)

                if len(ranking) >= limit:
                    break

            rankings.append((ranking, num_words))

            logger.debug(f"num_words : {num_words}")
            logger.debug(f"word_ids : {word_ids}")
            logger.debug(f"limit : {limit, rank_limit, num_words}")
            logger.debug(f"pred: {pred}")
            logger.debug(f"ranking: {ranking}")

        return rankings

    @staticmethod
    def ordering2ranks(
        orderings: list[tuple[list[int], int]], rank_limit: float | int
    ) -> list[list[int]]:
        ranks = []

        for batch_idx, ordering_tuple in enumerate(orderings):
            ordering = ordering_tuple[0]
            num_words = ordering_tuple[1]
            rank = np.ones(num_words) * (len(ordering) + 1)  # first rank = 1

            for i, pos in enumerate(ordering):
                rank[pos] = i + 1  # first rank = 1

            ranks.append(rank)

            logger.debug(f"ordering_tuple : {ordering_tuple}")
            logger.debug(f"rank ones : {rank}")

        return ranks

    def logits2ranks(
        self,
        logits: torch.Tensor,
        tokenized_inputs: BatchEncoding,
        rank_limit: float | int,
    ) -> list[list[int]]:
        logits = torch.softmax(logits, dim=-1)
        logits = logits * tokenized_inputs.data["attention_mask"][:, :, None]
        logits = logits[:, :, 0]  # keep prob of not inserted

        ignore_mask = EvalWordImp.get_ignore_mask(
            tokenized_inputs, self.prohibited_word_ids
        )
        logits = logits * (1 - ignore_mask)  # mask ignore tokens
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        orderings = EvalWordImp.sorted2ordering(
            sorted_indices, tokenized_inputs, rank_limit
        )
        ranks = EvalWordImp.ordering2ranks(orderings, rank_limit)

        logger.debug(f"sorted_indices : {sorted_indices}")
        logger.debug(f"orderings : {orderings}")
        logger.debug(f"ignore_mask: {ignore_mask}")

        return ranks

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        data_dir = os.path.join("./data/eval/", self.job_name)
        os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        with open(os.path.join(data_dir, "config.json"), "w") as file:
            file.write(config)
        logger.info(f"Started Word Importance evaluation job with this args:\n{config}")

        ds = load_ds()
        logger.info(ds)

        self.load_model()
        tokenized_inputs = self.tokenize(ds)
        logits = self.predict(tokenized_inputs)
        ranks = self.logits2ranks(
            logits, tokenized_inputs, rank_limit=self.max_rank_limit
        )

        labels = ds["label"]
        labels = [rankdata(label, method="ordinal") for label in labels]

        limit = self.max_rank_limit
        logger.debug("=======ranks=========")
        logger.debug(ranks[0])
        ranks = RankingEvaluator.ignore_maximal(ranks, rank_limit=limit)
        logger.debug(ranks[0])

        logger.debug("=======labels=========")
        logger.debug(labels[0])
        labels = RankingEvaluator.ignore_maximal(labels, rank_limit=limit)
        logger.debug(labels[0])

        results = {}
        results["name"] = get_model_name(self.hf_model)
        logger.info(f"model : {results['name']}")

        results["pearson"], p_vals = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "pearson"
        )
        results["pearson_pvalue"] = p_vals[0]
        logger.info(f"pearson : {results['pearson']}")

        results["kendall"], p_vals = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "kendall"
        )
        results["kendall_pvalue"] = p_vals[0]
        logger.info(f"kendal : {results['kendall']}")

        results["somers"] = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "somers"
        )
        logger.info(f"sommer : {results['somers']}")

        for k in range(1, 6):
            k_inter = f"{k}-inter"
            results[k_inter] = RankingEvaluator.least_intersection(ranks, labels, k)
            logger.info(f"{k_inter} : {results[k_inter]}")

        results["avg_overlap"] = RankingEvaluator.avg_overlaps(ranks, labels, limit)
        logger.info(f"avg_overlap : {results['avg_overlap']}")

        with open(os.path.join(data_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        return


@dataclass
class EvalWordImpTFIDF(ConfigurableJob):
    seed: int = 69
    max_rank_limit: int | float = 0.1

    job_version: str = "0.1"

    @staticmethod
    def get_terms(contexts: list[list[str]]) -> dict[str, int]:
        terms = set(term for doc in contexts for term in doc)
        terms = {term: i for i, term in enumerate(terms)}
        return terms

    @staticmethod
    def get_idf(contexts: list[list[str]], terms: dict[str, int]) -> np.ndarray:
        idf = np.zeros(len(terms))
        for doc in contexts:
            for term in set(doc):
                idf[terms[term]] += 1
        idf = np.log(len(contexts) / idf)
        return idf

    @staticmethod
    def get_tf_idf(contexts: list[list[str]]) -> list[np.ndarray]:
        terms = EvalWordImpTFIDF.get_terms(contexts)
        idf = EvalWordImpTFIDF.get_idf(contexts, terms)

        tfidf_weights = []

        for doc in contexts:
            counts = Counter(doc)
            tfidf = np.zeros(len(doc))
            for i, w in enumerate(doc):
                if not is_prohibited_word(w):
                    tfidf[i] = (counts[w] / len(doc)) * idf[terms[w]]
            tfidf_weights.append(tfidf)

        return tfidf_weights

    def tfidf2ranks(self, tfidf: list[np.ndarray]) -> list[list[int]]:
        sorted_words = [np.argsort(c)[::-1] for c in tfidf]
        ranks = []

        for sorted_doc_words in sorted_words:
            limit = get_rank_limit(self.max_rank_limit, len(sorted_doc_words))
            rank = np.ones(len(sorted_doc_words)) * (limit + 1)

            for i, pos in enumerate(sorted_doc_words[:limit]):
                rank[pos] = i + 1

            ranks.append(rank)

        return ranks

    def run(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        data_dir = os.path.join("./data/eval/", self.job_name)
        os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        with open(os.path.join(data_dir, "config.json"), "w") as file:
            file.write(config)
        logger.info(
            f"Started TFIDF Word Importance evaluation job with this args:\n{config}"
        )

        ds = load_ds()
        contexts = [[w.lower() for w in context] for context in ds["context"]]

        labels = ds["label"]
        labels = [rankdata(label, method="ordinal") for label in labels]
        labels = RankingEvaluator.ignore_maximal(labels, rank_limit=self.max_rank_limit)

        tfidf = EvalWordImpTFIDF.get_tf_idf(contexts)
        ranks = self.tfidf2ranks(tfidf)

        results = {}
        results["name"] = "tf-idf"
        logger.info(f"model : {results['name']}")

        results["pearson"], p_vals = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "pearson"
        )
        results["pearson_pvalue"] = p_vals[0]
        logger.info(f"pearson : {results['pearson']}")

        results["kendall"], p_vals = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "kendall"
        )
        results["kendall_pvalue"] = p_vals[0]
        logger.info(f"kendal : {results['kendall']}")

        results["somers"] = RankingEvaluator.mean_rank_correlation(
            ranks, labels, "somers"
        )
        logger.info(f"sommer : {results['somers']}")

        for k in range(1, 6):
            k_inter = f"{k}-inter"
            results[k_inter] = RankingEvaluator.least_intersection(ranks, labels, k)
            logger.info(f"{k_inter} : {results[k_inter]}")

        results["avg_overlap"] = RankingEvaluator.avg_overlaps(
            ranks, labels, self.max_rank_limit
        )
        logger.info(f"avg_overlap : {results['avg_overlap']}")

        with open(os.path.join(data_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        self.results = results

        return
