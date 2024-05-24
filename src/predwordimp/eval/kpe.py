import json
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from datasets import load_dataset
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from nltk.stem.snowball import SnowballStemmer as Stemmer
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils import BatchEncoding

from predwordimp.eval.wi_eval import EvalWordImpTFIDF
from predwordimp.util.job import ConfigurableJob
from predwordimp.util.logger import get_logger

logger = get_logger(__name__)

# TODO add creation of json predictions for understanding


@dataclass
class Candidate:
    """Represents a candidate phrase for keyphrase extraction.

    Attributes:
        phrase (str): The candidate phrase.
        start_pos (int): The starting position of the phrase in the original text.
        end_pos (int): The ending position of the phrase in the original text.
    """

    phrase: str
    start_pos: int
    end_pos: int


@dataclass
class KpeEvalJob(ConfigurableJob):
    """Job for evaluating keyphrase extraction models. Our models and TF-IDF.

    Attributes:
        model_names_or_paths (list[str]): A list of HF model names or paths to evaluate.
        dataset (str): The name of the dataset to use for evaluation.
        split (str): The split of the dataset to use (default is "test").
        dataset_subset (str): The subset of the dataset to use (default is "generation").
            Useful when the HF dataset has extractive and generation keyphrase mode.
        output_dir_name (str | None): The name of the output directory for results. (default is None).
    """

    model_names_or_paths: list[str]
    dataset: Literal[
        "taln-ls2n/inspec",
        "taln-ls2n/semeval-2010-pre",
        "midas/duc2001",
        "midas/semeval2017",
    ]
    split: str = "test"
    dataset_subset: str = "generation"
    output_dir_name: str | None = None
    predictions = None

    @staticmethod
    def get_candidate_pos(text: str) -> list[Candidate]:
        """Extracts candidate phrases from the text using noun phrase grammar.

        Args:
            text (str): The input text.

        Returns:
            list[Candidate]: A list of candidate phrases with their positions.
        """
        noun_phrases = []
        grammar = r"""
    NP: {<JJ>*<NN|NNS|NNP|NNPS>+}  # Adjectives (zero or more) followed by nouns (one or more)
    """
        cp = RegexpParser(grammar)
        char_pos = 0
        words = word_tokenize(text)
        # words = text.split()
        tagged_words = pos_tag(words)
        tree = cp.parse(tagged_words)

        for subtree in tree.subtrees():  # type: ignore
            if subtree.label() == "NP":
                phrase = " ".join(word for word, pos in subtree.leaves())
                start_pos = text.find(phrase, char_pos)
                if start_pos == -1:
                    logger.info(f"Phrase not found in the text! The phrase `{phrase}`")
                    continue
                end_pos = start_pos + len(phrase)
                noun_phrases.append(Candidate(phrase, start_pos, end_pos))
                char_pos = end_pos

        return noun_phrases

    @staticmethod
    def eval_prf(
        top_N_keyphrases: list[str], references: list[str], cutoff: int = 5
    ) -> tuple[float, float, float]:
        """Evaluates precision, recall, and F1-score for the top N ranked keyphrases.

        Args:
            top_N_keyphrases (list[str]): A list of top N keyphrases.
            references (list[str]): A list of reference keyphrases.
            cutoff (int): The cutoff value for evaluation (default is 5).

        Returns:
            tuple[float, float, float]: The precision, recall, and F1-score.
        """
        P = (
            len(set(top_N_keyphrases[:cutoff]) & set(references))
            / len(top_N_keyphrases[:cutoff])
            if len(top_N_keyphrases) > 0
            else 1
        )
        R = len(set(top_N_keyphrases[:cutoff]) & set(references)) / len(set(references))
        F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
        return (P, R, F)

    def run(self) -> None:
        """Runs the keyphrase extraction evaluation job and saves the results as a json."""
        self.predictions = dict()
        data_dir = os.path.join(
            "./data/kpe_vis/",
            self.output_dir_name if self.output_dir_name else self.job_name,
        )
        os.makedirs(data_dir, exist_ok=True)

        config = self.get_config()
        logger.info(
            f"Started keyphrase extraction evaluation job with this args:\n{config}"
        )

        results = dict()
        for model in self.model_names_or_paths:
            name = model.split("/")[-1]
            results[name] = self.eval_model(model)
        results["dataset"] = self.dataset

        with open(os.path.join(data_dir, "results.json"), "w") as f:
            f.write(json.dumps(results, indent=4))
        
        with open(os.path.join(data_dir, "predictions.json"), "w") as f:
            f.write(json.dumps(self.predictions, indent=4))

    @staticmethod
    def weight_candidates(
        text: str, preds: Any, offsets: Any, tensor: bool
    ) -> list[tuple[str, float]]:
        """At first it extracts possible candidates according to `get_candidate_pos`.
        It computes the importance as a mazimum of the `preds` scores of the words in the phrase.
        Then it reorders the canditates according to this importance.

        Args:
            text (str): The input text.
            preds (Any): The model predictions.
            offsets (Any): The offsets of the tokens.
            tensor (bool): Whether the predictions are tensors.

        Returns:
            list[str]: A list of weighted candidate phrases.
        """
        candidates = KpeEvalJob.get_candidate_pos(text)
        skipped = 0
        candidate_importance = []
        for candidate in candidates:
            token_ids = [
                idx
                for idx, offset in enumerate(offsets)
                if offset[0] >= candidate.start_pos and offset[1] <= candidate.end_pos
            ]
            if token_ids == []:
                skipped += 1
                continue
            if tensor:
                importance = preds[0, token_ids, 0].max()  # [not inserted, inserted]
            else:
                importance = np.max(preds[token_ids])
            candidate_importance.append((candidate.phrase, importance))

        ranked_candidates = sorted(
            candidate_importance, key=lambda x: x[1], reverse=True
        )
        return ranked_candidates

    def load_dataset(self) -> tuple[list[str], list[str]]:
        """Loads the dataset for evaluation. It loads only extractive labels.

        Returns:
            tuple[list[str], list[str]]: A tuple containing the test texts and their corresponding labels.
        """
        if self.dataset in ["taln-ls2n/inspec", "taln-ls2n/semeval-2010-pre"]:
            dataset = load_dataset(
                self.dataset, split=self.split, trust_remote_code=True
            )
            dataset = dataset.select(range(3))

            test_texts = [
                (sample["title"] + ". " + sample["abstract"])  # type: ignore
                for sample in dataset
            ]
            text_phrases = dataset["keyphrases"]  # type: ignore
            prmu = dataset["prmu"]  # type: ignore
            labels = []
            for keyphrases, present in zip(text_phrases, prmu):
                extractive_keyphrases = [
                    p for i, p in enumerate(keyphrases) if present[i] == "P"
                ]
                labels.append(extractive_keyphrases)

        elif self.dataset in ["midas/duc2001", "midas/semeval2017"]:
            dataset = load_dataset(
                self.dataset, split=self.split, name=self.dataset_subset
            )
            dataset = dataset.select(range(3))

            test_texts = [" ".join(sample["document"]) for sample in dataset]  # type: ignore
            labels = dataset["extractive_keyphrases"]  # type: ignore

        return test_texts, labels

    def align_preds(self,
        predicted, labels: list[str], model_name_or_path
    ) -> tuple[list[list[str]], list[list[str]]]:
        """Aligns predicted keyphrases with reference keyphrases with stemming and casing.

        Args:
            predicted: The predicted keyphrases.
            labels (list[str]): The reference keyphrases.

        Returns:
            tuple[list[list[str]], list[list[str]]]: The aligned model outputs and references.
        """
        model_outputs = []
        references = []
        self.predictions[model_name_or_path] = dict()
        for i, prediction in enumerate(predicted):
            self.predictions[model_name_or_path][i] = dict()
            self.predictions[model_name_or_path][i]["pred"] = []
            self.predictions[model_name_or_path][i]["reference"] = []
            model_output = []
            for keyphrase, wi in prediction:
                toks = keyphrase.lower().split()
                model_output.append(
                    " ".join([Stemmer("english").stem(tok) for tok in toks])
                )
                self.predictions[model_name_or_path][i]["pred"].append((keyphrase, wi.item()))
            model_outputs.append(model_output)

            reference = []
            for keyphrase in labels[i]:
                toks = keyphrase.lower().split()
                reference.append(
                    " ".join([Stemmer("english").stem(tok) for tok in toks])
                )
                self.predictions[model_name_or_path][i]["reference"].append(keyphrase)
            references.append(reference)

        return model_outputs, references

    @staticmethod
    def eval_dict(
        model_outputs: list[list[str]], references: list[list[str]]
    ) -> dict[str, float]:
        """Evaluates the model outputs and returns precision, recall, and F1-scores.

        Args:
            model_outputs (list[list[str]]): The model outputs.
            references (list[list[str]]): The reference keyphrases.

        Returns:
            dict[str, float]: A dictionary containing the evaluation metrics.
        """
        results = dict()
        for cutoff in [5, 10, 15]:
            scores = []
            for i, output in enumerate(model_outputs):
                if len(references[i]) == 0:  # if no extractive keyphrase then skip
                    continue
                scores.append(KpeEvalJob.eval_prf(output, references[i], cutoff))

            P, R, F = np.mean(scores, axis=0)
            results[cutoff] = {
                "P": P,
                "R": R,
                "F": F,
            }
        return results

    @staticmethod
    def tokenize(
        tokenizer: PreTrainedTokenizerBase, text: str
    ) -> tuple[BatchEncoding, torch.Tensor]:
        """Tokenizes the input text using the provided tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
            text (str): The input text.

        Returns:
            tuple[BatchEncoding, torch.Tensor]: The tokenized input and offsets.
        """
        tokenized_inp = tokenizer(
            [text],
            truncation=True,
            padding=True,
            # return_overflowing_tokens=True,
            return_offsets_mapping=True,
            stride=128,
            is_split_into_words=True,
            return_tensors="pt",
        )
        offsets = tokenized_inp.pop("offset_mapping")[0]
        if tokenized_inp["input_ids"].shape[0] > 1:  # type: ignore
            raise ValueError("Too long input text for the context size.")

        return tokenized_inp, offsets

    @staticmethod
    def load_model_tokenizer(
        model_name_or_path,
    ) -> tuple[BertForTokenClassification | None, PreTrainedTokenizerBase]:
        """Loads the model and tokenizer from the given path.

        Args:
            model_name_or_path: The path to the model or model name.

        Returns:
            tuple[BertForTokenClassification | None, PreTrainedTokenizerBase]: The loaded model and tokenizer.
        """
        if model_name_or_path == "tf_idf":
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)

        return model, tokenizer

    def eval_model(self, model_name_or_path) -> dict[str, float]:
        """Evaluates a model for keyphrase extraction.

        Args:
            model_name_or_path: The path to the model or model name.

        Returns:
            dict[str, float]: The evaluation metrics.
        """
        model, tokenizer = KpeEvalJob.load_model_tokenizer(model_name_or_path)

        test_texts, labels = self.load_dataset()

        predicted = []

        all_tokenized_inp = []
        all_offsets = []
        token_contexts = []
        for text in test_texts:
            tokenized_inp, offsets = KpeEvalJob.tokenize(tokenizer, text)
            all_tokenized_inp.append(tokenized_inp)
            all_offsets.append(offsets)
            token_contexts.append(
                tokenizer.convert_ids_to_tokens(tokenized_inp["input_ids"][0])  # type: ignore
            )

        if model is None:
            tfidf_scores = EvalWordImpTFIDF.get_tf_idf(token_contexts)

        pbar = tqdm(total=len(all_tokenized_inp))
        for i, (tokenized_inp, offsets) in enumerate(
            zip(all_tokenized_inp, all_offsets)
        ):
            if model:
                tensor = True
                with torch.no_grad():
                    preds = model(**tokenized_inp).logits
                    preds = torch.softmax(preds, dim=-1)
            else:
                tensor = False
                preds = tfidf_scores[i]

            predicted.append(
                KpeEvalJob.weight_candidates(
                    test_texts[i], preds, offsets, tensor=tensor
                )
            )
            pbar.update()

        model_outputs, references = self.align_preds(predicted, labels, model_name_or_path)

        return KpeEvalJob.eval_dict(model_outputs, references)
