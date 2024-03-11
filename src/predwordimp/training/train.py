import os
from dataclasses import dataclass
from typing import ClassVar

import evaluate
import numpy as np
import torch
import transformers
from datasets import Dataset, load_dataset
from evaluate import EvaluationModule
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BatchEncoding,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

MASK_VALUE = -100


@dataclass
class TrainJob(ConfigurableJob):
    model: str
    dataset_dir: str

    seed: int = 69
    job_version: str = "0.2"

    num_proc: int = 0
    hf_access_token: str | None = None
    lr: float = 2e-5
    batch_size: int = 64
    epochs: int = 1
    wd: float = 0.0
    warmup_steps: int = 1000
    stride: int = 128

    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500

    seqeval: ClassVar[EvaluationModule] = evaluate.load("seqeval")
    id2label: ClassVar[dict[int, str]] = {0: "not inserted", 1: "inserted"}
    label2id: ClassVar[dict[str, int]] = {"not inserted": 0, "inserted": 1}
    label_list: ClassVar[list[str]] = ["not inserted", "inserted"]

    @classmethod
    def compute_metrics(cls, p) -> dict[str, float]:
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [cls.label_list[p] for (p, l) in zip(prediction, label) if l != MASK_VALUE]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [cls.label_list[l] for (p, l) in zip(prediction, label) if l != MASK_VALUE]
            for prediction, label in zip(predictions, labels)
        ]

        results = cls.seqeval.compute(
            predictions=true_predictions, references=true_labels
        )

        if results is None:
            raise RuntimeError("Computation of metrics in seqeval returned None.")

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def tokenize_and_align_labels(self, samples) -> BatchEncoding:
        tokenized_inputs = self.tokenizer(
            samples["words"],
            truncation=True,
            return_overflowing_tokens=True,
            stride=self.stride,
            is_split_into_words=True,
            return_tensors="np",
        )

        labels = []
        for i in range(tokenized_inputs["overflow_to_sample_mapping"].shape[0]):
            sample_idx = tokenized_inputs["overflow_to_sample_mapping"][i]
            label = samples["target"][sample_idx]

            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to MASK_VALUE.
                if word_idx is None:
                    label_ids.append(MASK_VALUE)
                elif (
                    word_idx != previous_word_idx
                ):  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(MASK_VALUE)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def load_datasets(self) -> tuple[Dataset, Dataset]:
        train_ds = load_dataset(
            "json",
            data_files=os.path.join(self.dataset_dir, "train.jsonl"),
            split="train",
        )
        train_ds = train_ds.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=train_ds.column_names,
        )

        val_ds = load_dataset(
            "json",
            data_files=os.path.join(self.dataset_dir, "validation.jsonl"),
            split="train",
        )
        val_ds = val_ds.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=val_ds.column_names,
        )

        return train_ds, val_ds  # type: ignore

    def run(self) -> None:
        np.random.seed(self.seed)
        transformers.set_seed(self.seed)
        logger = get_logger(__name__)
        data_dir = os.path.join("./data/train/", self.job_name)
        os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        with open(os.path.join(data_dir, "train_config.json"), "w") as file:
            file.write(config)
        logger.info(f"Started WikiText dataset creation job with this args:\n{config}")

        logger.info(f"Loading and preprocessing dataset {self.dataset_dir}")
        self.load_tokenizer()
        train_ds, val_ds = self.load_datasets()

        logger.info(f"Loading model {self.model}.")
        model = AutoModelForTokenClassification.from_pretrained(
            self.model,
            num_labels=2,
            id2label=TrainJob.id2label,
            label2id=TrainJob.label2id,
        )

        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU: {torch.cuda.get_device_properties(i)}")

        training_args = TrainingArguments(
            output_dir=os.path.join(data_dir, self.job_name),
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.wd,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=1,
            # load_best_model_at_end=True,
            hub_token=self.hf_access_token,
            warmup_steps=self.warmup_steps,
            seed=self.seed,
            dataloader_num_workers=self.num_proc,
            run_name=self.job_name,
            logging_dir=os.path.join(data_dir, "tensorboard"),
            logging_strategy="steps",
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            logging_steps=self.logging_steps,
        )

        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer
        )  # dynamically pad the inputs

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=TrainJob.compute_metrics,
        )

        trainer.train()

        if self.hf_access_token:
            trainer.push_to_hub()
