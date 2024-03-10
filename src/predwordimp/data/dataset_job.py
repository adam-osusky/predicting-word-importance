import concurrent.futures
import json
import os
import random
import re
from dataclasses import dataclass
from logging import Logger
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import (
    Dataset,
    load_dataset,
)
from mosestokenizer import MosesDetokenizer
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

# for unit tests and debugging
test_len = 100
start = 100  # for case with sample longer than 512
test_range = range(start, start + test_len)


def load_wiki_ds(split: str) -> Dataset:
    dataset = load_dataset(
        path="wikitext",
        name="wikitext-103-raw-v1",
        split=split,
        streaming=False,
    )
    return dataset  # type: ignore


@dataclass
class WikiTextDsJob(ConfigurableJob):
    seed: int = 69
    num_proc: int | None = None
    insert_rate: float = 0.5
    insert_model: str = "random"
    max_size: int | None = None
    job_version: str = "1.1"
    debug: bool = False

    @staticmethod
    def merge_intratokens(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Moses tokenizer tokenizes intra special characters with @<special-char>@ tagging."""
        if len(sample["text"]) != 0:
            sample["text"] = re.sub(
                pattern=r" @(.)@ ",
                repl=lambda match: match.group(1),
                string=sample["text"],
            )
        return sample

    @staticmethod
    def detokenize(sample: Dict[str, Any]) -> Dict[str, Any]:
        with MosesDetokenizer("en") as detokenize:
            sample["text"] = detokenize(sample["text"].split())
        return sample

    def preprocess_dataset(self, ds: Dataset) -> Dataset:
        """Remove samples of empty lines and titles."""
        ds = ds.filter(
            function=lambda sample: sample["text"] != ""
            and not sample["text"].startswith(" = "),
            num_proc=self.num_proc,
        )
        ds = ds.map(function=WikiTextDsJob.merge_intratokens, num_proc=self.num_proc)
        ds = ds.map(function=WikiTextDsJob.detokenize, num_proc=self.num_proc)
        return ds

    def get_random_word(self) -> str:
        """Select uniformly random word from corpus."""

        if not self.full_ds:
            raise RuntimeError("For random word insertion is needed full dataset.")

        words = []
        while len(words) == 0:
            sample = self.full_ds[random.randint(a=0, b=len(self.full_ds) - 1)]
            words = sample["text"].split()

        return random.choice(seq=words)

    def insert_words(self, sample: Dict[str, Any]) -> dict[str, Any]:
        """Insert new words into the sample according to insertion method."""

        insertions_targets = self.get_insertions_targets(sample)

        if self.insert_model == "random":
            insertions_targets.pop("insert_positions")
            return insertions_targets

        else:
            filled_new_words = self.fill_mask(
                insertions_targets["words"], insertions_targets.pop("insert_positions")
            )
            insertions_targets["words"] = filled_new_words
            return insertions_targets

    def fill_mask(self, words: List[str], insert_positions: set[int]) -> List[str]:
        """Fill-mask task.
        For list of strings (words delimitted by whitespace) use lm model to predict mask
        tokens and fill the predictions.

        For prediction ignore neighbouring tokens. Example: [token1, MASK_TOKEN ,token2] ->
        from predictions removes token1 and token2.

        Also intra tokens (such as ##n, ##ion...) are ignored from predictions.

        In case of the tokenized sequence longer than model's max, the sequence will be
        splitted with no strides.
        """

        inputs = self.lm_tokenizer(
            words,
            return_tensors="pt",
            is_split_into_words=True,
            truncation=True,
            return_overflowing_tokens=True,
            padding="longest",
        )
        inputs.pop("overflow_to_sample_mapping", None)

        masked_idxs = torch.where(
            inputs["input_ids"] == self.lm_tokenizer.mask_token_id
        )

        logits = self.lm(**inputs).logits
        fill_logits = logits[masked_idxs]
        fill_logits[:, self.intra_word_mask] = -float("inf")  # ignore intra word tokens

        stride_idxs = masked_idxs[0]
        stride_masked_idxs = masked_idxs[1]
        neighbor_indices = [
            stride_masked_idxs - 1,
            stride_masked_idxs + 1,
        ]  # BOS and EOS so no need to worry out of bounds

        # ignore neighbouring tokens
        for i in range(len(neighbor_indices)):
            neighbours = neighbor_indices[i]
            neighbours_token_ids = inputs["input_ids"][stride_idxs, neighbours]
            fill_logits[:, neighbours_token_ids] = -float("inf")

        preds = torch.argmax(fill_logits, dim=1)
        word_preds = self.lm_tokenizer.convert_ids_to_tokens(preds)

        for insert_idx, txt_idx in enumerate(insert_positions):
            words[txt_idx] = word_preds[insert_idx]

        return words

    def load_lm(self) -> None:
        self.lm_tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(self.insert_model)
        )
        self.lm = AutoModelForMaskedLM.from_pretrained(self.insert_model)

        # get mask for intra-word tokens
        vocab = self.lm_tokenizer.get_vocab()
        intra_word_tokens = [
            token for token in vocab.keys() if token.startswith("##")
        ]  # specific to bert tokenizers

        vocab_size = self.lm_tokenizer.vocab_size
        self.intra_word_mask = [
            token in intra_word_tokens
            for token in self.lm_tokenizer.convert_ids_to_tokens(
                list(range(vocab_size))
            )
        ]

    def insert_str(self) -> str:
        """What string to insert at first in the insertion process."""

        if self.insert_model == "random":
            return self.get_random_word()
        else:
            return self.lm_tokenizer.mask_token

    def get_insertions_targets(self, sample: Dict[str, Any]) -> dict[str, Any]:
        """
        Helper function for insertion. Uniformly selects positions for insertions
        without repetition. Positions for insertions are spaces before original words.
        So consecutive insertion is not possible. Also one extra position for the end
        of the text.

        From the text is created list of words by whitespace spliting.
        """

        if len(sample["text"]) == 0:
            raise RuntimeError("An empty sample for insertion!")

        words = sample["text"].split()
        num_words = len(words)
        num_to_insert = int(num_words * self.insert_rate)

        insert_idxs = np.random.choice(
            a=range(num_words + 1), size=num_to_insert, replace=False
        )

        insertions = {}
        for i in insert_idxs:
            insertions[i] = self.insert_str()

        targets = [0] * (num_words + num_to_insert)

        inserted = 0
        new_words = []
        insert_positions = set()

        # insert word before original selcted word position
        for i, w in enumerate(words):
            if i in insertions:
                new_words.append(insertions[i])
                idx = i + inserted
                targets[idx] = 1
                insert_positions.add(idx)
                inserted += 1
            new_words.append(w)

        # insertion at the end
        i = num_words
        if i in insertions:
            new_words.append(insertions[i])
            idx = i + inserted
            targets[idx] = 1
            insert_positions.add(idx)
            inserted += 1

        new_sample = {
            "words": new_words,
            "target": targets,
            "insert_positions": insert_positions,
        }

        return new_sample

    def load_full_ds(self, log: Logger) -> None:
        log.info("Loading full dataset for random words.")
        self.full_ds = load_wiki_ds("all")
        if self.debug:
            self.full_ds = self.full_ds.select(test_range)

        log.info("Preprocessing the dataset for random words")
        self.full_ds = self.preprocess_dataset(self.full_ds)

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger = get_logger(__name__)
        data_dir = os.path.join("./data/wikitext/", self.job_name)
        os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        with open(os.path.join(data_dir, "config.json"), "w") as file:
            file.write(config)
        logger.info(f"Started WikiText dataset creation job with this args:\n{config}")

        if self.insert_model != "random":
            logger.info(
                f"Words will be inserted with {self.insert_model} language model."
            )
            logger.info("Loading language model and tokenizer.")
            self.load_lm()
            self.full_ds = None
        else:
            logger.info("Words will be inserted randomly from corpus.")
            self.lm = None
            self.load_full_ds(logger)

        for splt in ["train", "validation", "test"]:
            logger.info(f"Loading the {splt} part of the dataset.")
            dataset = load_wiki_ds(splt)
            if self.debug:
                dataset = dataset.select(test_range)

            logger.info(f"Preprocessing the {splt} part of the dataset.")
            dataset = self.preprocess_dataset(dataset)

            logger.info(f"Started insertion process for the {splt} part.")
            count = 0
            with open(os.path.join(data_dir, f"{splt}.jsonl"), "w") as jsn:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_proc
                ) as executor:
                    futures = [
                        executor.submit(self.insert_words, sample) for sample in dataset
                    ]

                    for future in tqdm(concurrent.futures.as_completed(futures)):
                        try:
                            example = future.result()
                            json.dump(example, jsn)
                            jsn.write("\n")  # jsonLINES
                            count += 1
                            if self.max_size and count >= self.max_size:
                                break
                        except Exception as e:
                            logger.error(f"Error during word insertion: {e}")

            logger.info(f"Finished insertion process for the {splt} part.")

        return
