from dataclasses import dataclass
import json
from typing import Any, Literal
from tqdm import tqdm

import nltk
import numpy as np
import torch
from datasets import load_dataset
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from nltk.stem.snowball import SnowballStemmer as Stemmer
from transformers import AutoModelForTokenClassification, AutoTokenizer, BertForTokenClassification, PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer, BatchEncoding
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from predwordimp.eval.wi_eval import EvalWordImpTFIDF
from predwordimp.util.job import ConfigurableJob


@dataclass
class Candidate:
    phrase: str
    start_pos: int
    end_pos: int


@dataclass
class KpeEvalJob(ConfigurableJob):
    model_names_or_paths: list[str]
    dataset: str
    split: str = "test"
    dataset_subset: str = "generation"

    @staticmethod
    def get_candidate_pos(text: str) -> list[Candidate]:
        noun_phrases = []
        #     grammar = r"""
        # NP: {<DT>?<JJ>*<NN.*>+}   # Chunk sequences of DT, JJ, NN
        #     {<NNP>+}            # Chunk sequences of NNP (Proper Nouns)
        #     {<NN.*><NN.*>}          # Chunk sequences of NN NN (e.g., 'computer science')
        #     {<DT>?<JJ>*<NN.*>+<CC><NN.*>+}    # Noun Phrases with Coordinating Conjunctions
        #     {<NNP><NNP><NNP>*}            #(Sequences of Proper Nouns)
        # """
        grammar = r"""
    NP: {<JJ>*<NN|NNS|NNP|NNPS>+}  # Adjectives (zero or more) followed by nouns (one or more)
    """
        cp = RegexpParser(grammar)
        char_pos = 0
        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        tree = cp.parse(tagged_words)

        for subtree in tree.subtrees(): # type: ignore
            if subtree.label() == "NP":
                phrase = " ".join(word for word, pos in subtree.leaves())
                start_pos = text.find(phrase, char_pos)
                if start_pos == -1:
                    # raise ValueError("Phrase not found in the text!")
                    print("Phrase not found in the text!")
                    continue
                end_pos = start_pos + len(phrase)
                noun_phrases.append(Candidate(phrase, start_pos, end_pos))
                char_pos = end_pos

        return noun_phrases

    @staticmethod
    def eval_prf(top_N_keyphrases: list[str], references: list[str], cutoff: int = 5) -> tuple[float, float, float]:
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
        results = dict()
        for model in self.model_names_or_paths:
            name = model.split("/")[-1]
            results[name] = self.eval_model(model)
        
        print(results)
        print("="*10)
        json_string = json.dumps(results, indent=4)
        print(json_string)
    
    @staticmethod
    def weight_candidates(text: str, preds: Any, offsets: Any, tensor: bool) -> list[str]:
        candidates = KpeEvalJob.get_candidate_pos(text)
        skipped = 0
        candidate_importance = []
        for candidate in candidates:
            token_ids = [
                idx
                for idx, offset in enumerate(offsets)
                if offset[0] >= candidate.start_pos
                and offset[1] <= candidate.end_pos
            ]
            if token_ids == []:
                skipped += 1
                continue
            if tensor:
                importance = preds[0, token_ids, 0].mean()
            else:
                importance = np.mean(preds[token_ids])
            candidate_importance.append((candidate.phrase, importance))
        
        ranked_candidates = sorted(candidate_importance, key=lambda x: x[1])
        return [e[0] for e in ranked_candidates]
    
    def load_dataset(self) -> tuple[list[str], list[str]]:
        if self.dataset in ["taln-ls2n/inspec", "taln-ls2n/semeval-2010-pre"]:
            dataset = load_dataset(self.dataset, split=self.split)
            dataset = dataset.select(range(10))

            test_texts = [(sample["title"] + ". " + sample["abstract"]) for sample in dataset] # type: ignore
            text_phrases = dataset["keyphrases"] # type: ignore
            prmu = dataset["prmu"] # type: ignore
            labels = []
            for keyphrases, present in zip(text_phrases, prmu):
                extractive_keyphrases = [p for i, p in enumerate(keyphrases) if present[i] == "P"]
                labels.append(extractive_keyphrases)

        elif self.dataset in ["midas/duc2001", "midas/semeval2017"]:
            dataset = load_dataset(self.dataset, split=self.split, name=self.dataset_subset)
            dataset = dataset.select(range(10))

            test_texts = [" ".join(sample["document"]) for sample in dataset] # type: ignore
            labels = dataset["extractive_keyphrases"] # type: ignore
        
        return test_texts, labels
    
    @staticmethod
    def align_preds(predicted, labels: list[str]) -> tuple[list[list[str]], list[list[str]]]:
        model_outputs = []
        references = []
        for i, prediction in enumerate(predicted):
            model_output = []
            for keyphrase in prediction:
                toks = keyphrase.split()
                model_output.append(
                    " ".join([Stemmer("english").stem(tok) for tok in toks])
                )
            model_outputs.append(model_output)

            reference = []
            for keyphrase in labels[i]:
                toks = keyphrase.split()
                reference.append(
                    " ".join([Stemmer("english").stem(tok) for tok in toks])
                )
            references.append(reference)
        
        return model_outputs, references
    
    @staticmethod
    def eval_dict(model_outputs: list[list[str]], references: list[list[str]]) -> dict[str, float]:
        results = dict()
        for cutoff in [5, 10, 15]:
            scores = []
            for i, output in enumerate(model_outputs):
                scores.append(KpeEvalJob.eval_prf(output, references[i], cutoff))

            P, R, F = np.mean(scores, axis=0)
            results[cutoff] = {
                "P": P,
                "R": R,
                "F": F,
            }
        return results
    
    @staticmethod
    def tokenize(tokenizer: PreTrainedTokenizerBase, text: str) -> tuple[BatchEncoding, torch.Tensor]:
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
        if tokenized_inp["input_ids"].shape[0] > 1: # type: ignore
            raise ValueError("Too long inout text for the context size.")
        
        return tokenized_inp, offsets
    
    @staticmethod
    def load_model_tokenizer(model_name_or_path) -> tuple[BertForTokenClassification | None, PreTrainedTokenizerBase]:
        if model_name_or_path == "tf_idf":
            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
            model = None
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        
        return model, tokenizer
        

    def eval_model(self, model_name_or_path) -> dict[str, float]:
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
                tokenizer.convert_ids_to_tokens(tokenized_inp["input_ids"][0]) # type: ignore
            )
        
        if model is None:
            tfidf_scores = EvalWordImpTFIDF.get_tf_idf(token_contexts)
        
        pbar = tqdm(total=len(all_tokenized_inp))
        for i, (tokenized_inp, offsets) in enumerate(zip(all_tokenized_inp, all_offsets)):
            if model:
                tensor=True
                with torch.no_grad():
                    preds = model(**tokenized_inp).logits
                    preds = torch.softmax(preds, dim=-1)
            else:
                tensor=False
                preds = tfidf_scores[i]

            predicted.append(
                KpeEvalJob.weight_candidates(test_texts[i], preds, offsets, tensor=tensor)
            )
            pbar.update()

        model_outputs, references = KpeEvalJob.align_preds(predicted, labels)
        
        return KpeEvalJob.eval_dict(model_outputs, references)
