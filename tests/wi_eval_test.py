from typing import Any

import pytest
import torch
from predwordimp.eval.metrics import RankingEvaluator, rankings
from predwordimp.eval.util import get_rank_limit
from predwordimp.eval.wi_eval import EvalWordImp
from predwordimp.util.logger import get_logger
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding

logger = get_logger(__name__, log_level=20)


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
def rankings_ten() -> rankings:
    return [
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [5, 3, 5, 2, 5, 4, 5, 1, 5, 5],
        [2, 3, 3, 3, 3, 3, 3, 3, 1, 3],
    ]


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


@pytest.mark.parametrize("rank_limit", [1, 2, 3, 0.1, 0.25, 0.5, 0.75])
def test_logits2ranks(
    logits: torch.Tensor,
    tokenized_inputs: BatchEncoding,
    eval_job: EvalWordImp,
    ds: dict[str, Any],
    rank_limit: int | float,
) -> None:
    logger.debug(f"RANK LIMIT : {rank_limit}")
    ranks = eval_job.logits2ranks(logits, tokenized_inputs, rank_limit)
    logger.debug(f"ranks : {ranks}")

    logger.debug("==========")
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][0])
    logger.debug(tokens)
    tokens = tokenizer.convert_ids_to_tokens(tokenized_inputs["input_ids"][1])
    logger.debug(tokens)

    for i in range(len(ranks)):
        rank = ranks[i]
        context = ds["context"][i]
        logger.debug(rank)

        # first rank = 1 so rank for unselected is len(ordering) + 1
        max_rank = get_rank_limit(rank_limit, len(context)) + 1

        assert len(rank) == len(context)
        assert max(rank) <= len(context)
        assert min(rank) == 1  # first rank = 1
        assert max(rank) == max_rank

        for i in range(1, max_rank + 1):
            assert i in rank
        assert max_rank + 1 not in rank


@pytest.mark.parametrize("rank_limit", [1, 2, 3, 0.1, 0.2, 0.3])
def test_ignore_maximal(rankings_ten: rankings, rank_limit: int | float) -> None:
    ignored = RankingEvaluator.ignore_maximal(rankings_ten, rank_limit=rank_limit)

    for i in range(len(ignored)):
        row = ignored[i]
        max_selected_rank = get_rank_limit(rank_limit, len(row))

        assert max(row) == max_selected_rank + 1

        assert sorted(set(row), reverse=True)[1] <= max_selected_rank
        assert max_selected_rank + 2 not in row

        selected = sorted([r for r in row if r <= max_selected_rank])
        for j in range(len(selected) - 1):
            assert selected[j] + 1 == selected[j + 1]
