from dataclasses import dataclass

import nltk
import numpy as np
import torch
from datasets import load_dataset
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from nltk.stem.snowball import SnowballStemmer as Stemmer
from transformers import AutoModelForTokenClassification, AutoTokenizer

from predwordimp.util.job import ConfigurableJob


@dataclass
class Candidate:
    phrase: str
    start_pos: int
    end_pos: int


@dataclass
class KpeEvalJob(ConfigurableJob):
    model_name_or_path: str
    dataset: str

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

        for subtree in tree.subtrees():
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
    def evaluate(top_N_keyphrases, references, cutoff=5):
        P = len(set(top_N_keyphrases[:cutoff]) & set(references)) / len(
            top_N_keyphrases[:cutoff]
        ) if len(top_N_keyphrases) > 0 else 1
        R = len(set(top_N_keyphrases[:cutoff]) & set(references)) / len(set(references))
        F = (2 * P * R) / (P + R) if (P + R) > 0 else 0
        return (P, R, F)

    def run(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path)

        dataset = load_dataset(self.dataset, split="test")

        test_texts = [sample["title"] + ". " + sample["abstract"] for sample in dataset]

        predicted = []

        for i, text in enumerate(test_texts):
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
            tokens = tokenized_inp.tokens()
            if tokenized_inp["input_ids"].shape[0] > 1:
                raise ValueError("Too long inout text for the context size.")

            with torch.no_grad():
                preds = model(**tokenized_inp).logits
            preds = torch.softmax(preds, dim=-1)

            candidates = KpeEvalJob.get_candidate_pos(text)

            candidate_importance = []
            for candidate in candidates:
                token_ids = [
                    idx
                    for idx, offset in enumerate(offsets)
                    if offset[0] >= candidate.start_pos
                    and offset[1] <= candidate.end_pos
                ]
                importance = preds[0, token_ids, 0].max(dim=0)
                candidate_importance.append((candidate.phrase, importance))

            ranked_candidates = sorted(candidate_importance, key=lambda x: x[1])
            predicted.append([e[0] for e in ranked_candidates])

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
            for idx, keyphrase in enumerate(dataset[i]["keyphrases"]):
                # if dataset[i]["prmu"][idx] != "P":
                #     continue
                toks = keyphrase.split()
                reference.append(
                    " ".join([Stemmer("english").stem(tok) for tok in toks])
                )
            references.append(reference)

        for cutoff in [5, 10, 1000]:
            scores = []
            for i, output in enumerate(model_outputs):
                scores.append(KpeEvalJob.evaluate(output, references[i], cutoff))

            # compute the average scores
            P, R, F = np.mean(scores, axis=0)
            print(f"F@{cutoff}: F={F}, P={P}, R: {R}")
