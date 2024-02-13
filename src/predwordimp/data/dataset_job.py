import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from datasets import (
    Dataset,
    load_dataset,
)
from tqdm import tqdm

from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

# TODO: remove
test_len = 1000
start = 100
test_range = range(start, start + test_len)


def get_random_word(ds: Dataset) -> str:
    words = []
    while len(words) == 0:
        sample = ds[random.randint(a=0, b=len(ds) - 1)]
        words = sample["text"].split()
    return random.choice(seq=words)


@dataclass
class WikiTextDsJob(ConfigurableJob):
    seed: int = 69
    num_proc: int | None = None
    insert_rate: float = 0.5
    max_size: int | None = None
    job_version: str = "0.1"
    debug: bool = False

    @staticmethod
    def preprocess_text(sample: Dict[str, Any]) -> Dict[str, Any]:
        """Remove tokenization artefacts of moses tokenizer."""
        if len(sample["text"]) != 0:
            sample["text"] = re.sub(
                pattern=r"@(.)@",
                repl=lambda match: match.group(1),
                string=sample["text"],
            )
        return sample

    def preprocess_dataset(self, ds: Dataset) -> Dataset:
        """Remove samples of empty lines and titles."""
        ds = ds.filter(
            function=lambda sample: sample["text"] != ""
            and not sample["text"].startswith(" = "),
            num_proc=self.num_proc,
        )
        ds = ds.map(function=WikiTextDsJob.preprocess_text, num_proc=self.num_proc)
        return ds

    def insert_words(self, sample: Dict[str, Any], vocab_ds: Dataset) -> dict[str, Any]:
        words = sample["text"].split()
        num_words = len(words)
        num_to_insert = int(num_words * self.insert_rate)

        insert_idxs = np.random.choice(
            a=range(num_words + 1), size=num_to_insert, replace=False
        )

        insertions = {}
        for i in insert_idxs:
            insertions[i] = get_random_word(ds=vocab_ds)

        targets = [0] * (num_words + num_to_insert)

        inserted = 0
        new_words = []
        # insert word before original selcted word position
        for i, w in enumerate(words):
            # new_words.append(w)
            if i in insertions:
                new_words.append(insertions[i])
                targets[i + inserted] = 1
                inserted += 1
            new_words.append(w)

        # insertion at the end
        i = num_words
        if i in insertions:
            new_words.append(insertions[i])
            targets[i + inserted] = 1
            inserted += 1

        if len(new_words) == 0:
            raise RuntimeError(f"No new words inserted. /n{sample}")

        new_sample = {"words": new_words, "target": targets}
        return new_sample

    def run(self) -> None:
        np.random.seed(self.seed)
        random.seed(self.seed)

        logger = get_logger(__name__)
        data_dir = os.path.join("./data/wikitext/", self.job_name)
        os.makedirs(data_dir, exist_ok=True)
        print(__name__)
        logger.info(
            f"Started WikiText dataset creation job with this args:\n{self.get_config()}"
        )

        logger.info("Loading full dataset for random words.")
        self.full_ds = load_dataset(
            path="wikitext", name="wikitext-103-raw-v1", split="all", streaming=False
        )  # type: ignore
        if self.debug:
            self.full_ds = self.full_ds.select(test_range)  # type: ignore

        logger.info("Preprocessing the dataset for random words")
        self.full_ds = self.preprocess_dataset(self.full_ds)  # type: ignore

        for splt in ["train", "validation", "test"]:
            # for splt in ["train"]:

            logger.info(f"Loading the {splt} part of the dataset.")
            dataset = load_dataset(
                "wikitext", "wikitext-103-raw-v1", split=splt, streaming=False
            )
            if self.debug:
                dataset = dataset.select(test_range)  # type: ignore

            logger.info(f"Preprocessing the {splt} part of the dataset.")
            dataset = self.preprocess_dataset(dataset)

            logger.info(f"Started insertion process for the {splt} part.")
            count = 0
            with open(os.path.join(data_dir, f"{splt}.jsonl"), "w") as jsn:
                for sample in tqdm(iterable=dataset):
                    example = self.insert_words(sample, self.full_ds)

                    json.dump(example, jsn)
                    jsn.write("\n")  # jsonLINES
                    count += 1
                    if self.max_size and count >= self.max_size:
                        break

            logger.info(f"Finished insertion process for the {splt} part.")

        return
