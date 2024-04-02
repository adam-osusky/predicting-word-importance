from dataclasses import dataclass
import math
import os
import random

import numpy as np
import numpy.typing as npt
from datasets import load_dataset, Dataset
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger


@dataclass
class EvalWordImp(ConfigurableJob):
    hf_model: str
    seed: int = 69
    stride: int = 128

    job_version: str = "0.1"

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
            print(word_ids)
            limit = EvalWordImp.rank_limit(rank_limit, pred.shape[0])

            for index in pred:
                word_id = word_ids[index]

                if word_id is None:
                    continue

                ranking.append(word_id)

                if len(ranking) >= limit:
                    break

            rankings.append((ranking, max(wi for wi in word_ids if wi is not None) + 1))

        return rankings

    @staticmethod
    def ordering2ranks(orderings: list[tuple[list[int], int]]) -> list[list[int]]:
        ranks = []

        for batch_idx, ordering_tuple in enumerate(orderings):
            ordering = ordering_tuple[0]
            num_words = ordering_tuple[1]
            rank = np.ones(num_words) * len(ordering)

            for i, pos in enumerate(ordering):
                rank[pos] = i

            ranks.append(rank)

        return ranks

    def logits2ranks(
        self,
        logits: torch.Tensor,
        tokenized_inputs: BatchEncoding,
        rank_limit: float | int,
    ) -> torch.Tensor:
        logits = torch.softmax(logits, dim=-1)
        logits = logits * tokenized_inputs.data["attention_mask"][:, :, None]
        logits = logits[:, :, 0]  # keep prob of not inserted

        subword_mask = EvalWordImp.get_subwordmask(tokenized_inputs)
        logits = logits * (1 - subword_mask)  # mask subword tokens
        sorted_indices = torch.argsort(logits, dim=-1, descending=True)

        print(sorted_indices)

        orderings = EvalWordImp.sorted2ordering(
            sorted_indices, tokenized_inputs, rank_limit
        )
        print(orderings)
        ranks = EvalWordImp.ordering2ranks(orderings)

        return ranks

        return EvalWordImp.sorted2ordering(sorted_indices, tokenized_inputs, rank_limit)

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger = get_logger(__name__)
        data_dir = os.path.join("./data/evalwordimp/", self.job_name)
        # os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        # with open(os.path.join(data_dir, "config.json"), "w") as file:
        # file.write(config)
        logger.info(f"Started Word Importance evaluation job with this args:\n{config}")

        ds = self.load_ds()
        print(ds)

        self.load_model()

        tokenized_inputs = self.tokenize(ds)
        logits = self.predict(tokenized_inputs)

        ranks = self.logits2ranks(logits, tokenized_inputs)

        print(ranks[0])
        print(ranks[0][[53, 46, 62, 18, 22, 31, 5, 38, 42, 34]])

        tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        tokenized = tokenizer(
            [
                ["Adam", "is", "megafrajer", "."],
                [
                    "Guy",
                    "walks",
                    "into",
                    "the",
                    "doctor",
                    "'s",
                    "office",
                    "and",
                    "claims",
                ],
            ],
            truncation=True,
            padding=True,
            stride=128,
            is_split_into_words=True,
            return_tensors="pt",
        )
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])
        print(tokens)
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][1])
        print(tokens)

        ############

        # tokenizer = AutoTokenizer.from_pretrained(self.hf_model)
        # model = AutoModelForTokenClassification.from_pretrained(self.hf_model)

        # tokens = tokenizer(
        #     ds["context"],
        #     truncation=True,
        #     padding=True,
        #     return_overflowing_tokens=True,
        #     return_special_tokens_mask=True,
        #     stride=self.stride,
        #     is_split_into_words=True,
        #     return_tensors="pt",
        # )

        # print(tokens.keys())

        # overflow = tokens.pop("overflow_to_sample_mapping", None)
        # special_tokens_mask = tokens.pop("special_tokens_mask", None)

        # with torch.no_grad():
        #     out = model(**tokens).logits

        # print("out: ", out.shape)
        # print("attention masks: ", tokens["attention_mask"][:, :, None].shape)
        # out = torch.softmax(out, dim=-1)
        # out = out * tokens["attention_mask"][:, :, None]
        # out = out[:, :, 0]

        # subword_masks = []

        # # for batch_idx, pred in enumerate(out):
        # for batch_idx in range(tokens.data["input_ids"].shape[0]):
        #     word_ids = tokens.word_ids(batch_idx)
        #     previous_word_idx = None
        #     subword_mask = []
        #     for word_idx in word_ids:
        #         if word_idx is None:
        #             subword_mask.append(0.0)
        #         elif word_idx != previous_word_idx:
        #             subword_mask.append(0.0)
        #         else:
        #             subword_mask.append(1.0)
        #         previous_word_idx = word_idx
        #     subword_masks.append(subword_mask)

        # subword_mask = torch.tensor(subword_masks)
        # print("subword_mask: ", subword_mask.shape)

        # print(subword_mask[0])
        # for i, e in enumerate(subword_mask[0]):
        #     if e == 1:
        #         print(ds[0]["context"][tokens.word_ids(0)[i]], end=" ")
        # print()

        # out = out * (1 - subword_mask)

        # sorted_indices = torch.argsort(out, dim=-1, descending=True)

        # print("out transformed: ", out.shape)
        # print("sorted indices: ", sorted_indices.shape)

        # print(out[0])
        # print(sorted_indices[0])
        # print(out[0][sorted_indices[0]])

        # for batch_idx, pred in enumerate(sorted_indices):
        #     ranking = []
        #     repeated = []
        #     word_ids = tokens.word_ids(batch_idx)
        #     for index in pred:
        #         word_id = word_ids[index]
        #         if word_id is None:
        #             continue
        #         if word_id in ranking:
        #             repeated.append(ds[batch_idx]["context"][word_id])
        #         ranking.append(word_id)
        #         print(ds[batch_idx]["context"][word_id], end=" ")
        #     # print([word_ids[index] for index in pred])
        #     print(ranking)
        #     print(len(ranking), len(ds[batch_idx]["label"]))
        #     print(repeated)
        #     break

        return
