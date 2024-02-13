import pytest
from datasets import Dataset
from predwordimp.data.dataset_job import WikiTextDsJob


@pytest.fixture
def sample() -> dict[str, str]:
    return {
        "text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
    }


@pytest.fixture
def vocab_ds() -> Dataset:
    data = {
        "text": [
            "This is the first sentence.",
            "Here is the second sentence.",
            "Finally, the third sentence.",
        ]
    }

    return Dataset.from_dict(data)


@pytest.fixture
def wiki_text_ds_job() -> WikiTextDsJob:
    d = {
        "seed": 69,
    }
    return WikiTextDsJob.from_dict(d)


@pytest.mark.parametrize("insert_rate", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_insert_words(wiki_text_ds_job, sample, vocab_ds, insert_rate) -> None:
    wiki_text_ds_job.insert_rate = insert_rate

    result = wiki_text_ds_job.insert_words(sample, vocab_ds)

    assert "words" in result
    assert "target" in result
    assert isinstance(result["words"], list)
    assert isinstance(result["target"], list)

    num_words_sample = len(sample["text"].split())

    assert len(result["words"]) == len(result["target"])
    assert len(result["words"]) == num_words_sample + int(
        insert_rate * num_words_sample
    )

    without_inserted = []
    for w, t in zip(result["words"], result["target"]):
        if t == 0:
            without_inserted.append(w)

    without_inserted = " ".join(without_inserted)
    assert without_inserted == sample["text"]
