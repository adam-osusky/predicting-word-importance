import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from predwordimp.eval.metrics import RankingEvaluator
from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvalWordImp(ConfigurableJob):
    hf_model: str
    seed: int = 69
    stride: int = 128
    max_rank_limit: int | float = 0.1

    job_version: str = "0.2"

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        self.model = AutoModelForTokenClassification.from_pretrained(self.hf_model)

    def load_ds(self) -> Dataset:
        return load_dataset("adasgaleus/word-importance", split="test")  # type: ignore

    def tokenize(self, ds: Dataset) -> BatchEncoding:
        return self.tokenizer(
            ds["context"],
            truncation=True,
            padding=True,
            # return_overflowing_tokens=True,
            # return_special_tokens_mask=True,
            stride=self.stride,
            is_split_into_words=True,
            return_tensors="pt",
        )

    def predict(self, tokenized_inputs: BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            logits = self.model(**tokenized_inputs).logits
        return logits

    @staticmethod
    def get_subwordmask(tokenized_inputs: BatchEncoding):
        subword_masks = []

        for batch_idx in range(tokenized_inputs.data["input_ids"].shape[0]):
            word_ids = tokenized_inputs.word_ids(batch_idx)
            previous_word_idx = None
            subword_mask = []
            for word_idx in word_ids:
                if word_idx is None:
                    subword_mask.append(0)
                elif word_idx != previous_word_idx:
                    subword_mask.append(0)
                else:
                    subword_mask.append(1)
                previous_word_idx = word_idx
            subword_masks.append(subword_mask)

        return torch.tensor(subword_masks)

    @staticmethod
    def rank_limit(limit: int | float, length: int) -> int:
        if isinstance(limit, int):
            return limit
        elif isinstance(limit, float):
            return math.ceil(length * limit)

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
            limit = EvalWordImp.rank_limit(rank_limit, num_words)

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

        return rankings

    @staticmethod
    def ordering2ranks(
        orderings: list[tuple[list[int], int]], rank_limit: float | int
    ) -> list[list[int]]:
        ranks = []

        for batch_idx, ordering_tuple in enumerate(orderings):
            ordering = ordering_tuple[0]
            num_words = ordering_tuple[1]
            rank = np.ones(num_words) * len(ordering)

            for i, pos in enumerate(ordering):
                rank[pos] = i

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

        subword_mask = EvalWordImp.get_subwordmask(tokenized_inputs)
        logits = logits * (1 - subword_mask)  # mask subword tokens
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        orderings = EvalWordImp.sorted2ordering(
            sorted_indices, tokenized_inputs, rank_limit
        )
        ranks = EvalWordImp.ordering2ranks(orderings, rank_limit)

        logger.debug(f"sorted_indices : {sorted_indices}")
        logger.debug(f"orderings : {orderings}")

        return ranks

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        data_dir = os.path.join("./data/evalwordimp/", self.job_name)
        # os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        # with open(os.path.join(data_dir, "config.json"), "w") as file:
        # file.write(config)
        logger.info(f"Started Word Importance evaluation job with this args:\n{config}")

        ds = self.load_ds()
        logger.info(ds)

        self.load_model()
        tokenized_inputs = self.tokenize(ds)
        logits = self.predict(tokenized_inputs)
        ranks = self.logits2ranks(
            logits, tokenized_inputs, rank_limit=self.max_rank_limit
        )
        labels = ds["label"]

        spearman = RankingEvaluator.mean_rank_correlation(ranks, labels, "spearman")
        logger.info(f"spearman : {spearman}")

        kendal = RankingEvaluator.mean_rank_correlation(ranks, labels, "kendall")
        logger.info(f"kendal : {kendal}")

        sommer = RankingEvaluator.mean_rank_correlation(ranks, labels, "somers")
        logger.info(f"sommer : {sommer}")

        return
