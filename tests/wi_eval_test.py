import math
from typing import Any

import pytest
import torch
from predwordimp.eval.wi_eval import EvalWordImp
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


@pytest.fixture
def ds() -> dict[str, Any]:
    return {
        "context": [
            ["Adam", "is", "megafrajer", "."],
            ["Guy", "walks", "into", "the", "doctor", "'s", "office", "and", "claims"],
        ],
        "label": [
            [1, 2, 0, 2],
            [2, 3, 3, 3, 1, 3, 1.5, 3, 3],
        ],
    }


@pytest.fixture
def tokenized_inputs(ds: dict[str, Any]) -> BatchEncoding:
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer(
        ds["context"],
        truncation=True,
        padding=True,
        stride=128,
        is_split_into_words=True,
        return_tensors="pt",
    )


@pytest.fixture
def logits() -> torch.Tensor:
    torch.manual_seed(69)
    random_tensor = torch.rand([2, 12, 2])
    random_tensor = 2 * random_tensor - 1

    random_tensor[0][3][0] = 100.0
    random_tensor[0][3][1] = 0.0

    random_tensor[1][10][0] = 100.0
    random_tensor[1][10][1] = 0.0

    random_tensor[1][6][0] = 90.0
    random_tensor[1][6][1] = 0.0

    return random_tensor


@pytest.fixture
def eval_job() -> EvalWordImp:
    d = {
        "hf_model": "",
        "seed": 69,
        "stride": 128,
    }
    return EvalWordImp.from_dict(d)


def get_rank_limit(limit: int | float, length: int) -> int:
    if isinstance(limit, int):
        return limit
    elif isinstance(limit, float):
        return math.ceil(length * limit)


@pytest.mark.parametrize("rank_limit", [1, 2, 3, 0.1, 0.25, 0.5, 0.75])
def test_logits2ranks(
    logits: torch.Tensor,
    tokenized_inputs: BatchEncoding,
    eval_job: EvalWordImp,
    ds: dict[str, Any],
    rank_limit: int | float,
) -> None:
    print("RANK LIMIT :", rank_limit)
    ranks = eval_job.logits2ranks(logits, tokenized_inputs, rank_limit)
    print("ranks :", ranks)

    print("==========")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
    print(tokens)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][1])
    print(tokens)

    for i in range(len(ranks)):
        rank = ranks[i]
        context = ds["context"][i]
        print(rank)
        max_rank = get_rank_limit(rank_limit, len(context))
        assert len(rank) == len(context)
        assert max(rank) <= len(context)
        assert min(rank) == 0
        assert max(rank) == max_rank
        for i in range(max_rank + 1):
            assert i in rank
        assert max_rank + 1 not in rank
