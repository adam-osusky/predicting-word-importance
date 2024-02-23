import json
import os
from pathlib import Path

import pytest
from datasets import Dataset
from predwordimp.data.dataset_job import WikiTextDsJob, load_wiki_ds, test_range


@pytest.fixture
def sample() -> dict[str, str]:
    return {"text": "Over the years, she learned how to make a neat scarf."}


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
        "debug": True,
        "insert_rate": 0.5,
    }
    return WikiTextDsJob.from_dict(d)


@pytest.mark.parametrize("insert_rate", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("insert_method", ["random", "google-bert/bert-base-uncased"])
def test_insert_words(
    wiki_text_ds_job: WikiTextDsJob,
    sample: dict[str, str],
    vocab_ds: Dataset,
    insert_rate: float,
    insert_method: str,
) -> None:
    wiki_text_ds_job.insert_rate = insert_rate
    wiki_text_ds_job.insert_model = insert_method
    wiki_text_ds_job.full_ds = vocab_ds

    if insert_method != "random":
        wiki_text_ds_job.load_lm()

    result = wiki_text_ds_job.insert_words(sample)

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


@pytest.mark.parametrize("insert_method", ["random", "google-bert/bert-base-uncased"])
def test_run_wiki_text_ds_job(
    wiki_text_ds_job: WikiTextDsJob,
    tmp_path: Path,
    insert_method: str,
) -> None:
    job_name = "test_run_wiki_text_ds_job"
    wiki_text_ds_job.job_name = job_name
    data_dir = tmp_path / "data" / "wikitext" / job_name

    # cd to the temp directory
    os.chdir(str(tmp_path))

    # Run the job
    wiki_text_ds_job.insert_model = insert_method
    wiki_text_ds_job.run()

    config_file_path = data_dir / "config.json"
    assert config_file_path.is_file()

    for splt in ["train", "validation", "test"]:
        jsonl_file_path = data_dir / f"{splt}.jsonl"
        assert jsonl_file_path.is_file()

        # load og split to compare sequence of not inserted words
        ds = load_wiki_ds(splt)
        ds = ds.select(test_range)
        ds = wiki_text_ds_job.preprocess_dataset(ds)
        texts = set([sample["text"] for sample in ds])  # type: ignore

        with open(jsonl_file_path, "r") as jsonl_file:
            lines = jsonl_file.readlines()

            assert len(lines) == len(ds)  # for every sample insertion was completed

            for line in lines:
                example = json.loads(line)
                assert "words" in example
                assert "target" in example
                assert isinstance(example["words"], list)
                assert isinstance(example["target"], list)

                num_inserted = sum(example["target"])
                num_not_inserted = len(example["words"]) - num_inserted

                assert len(example["words"]) == len(example["target"])
                assert len(example["words"]) == num_not_inserted + int(
                    wiki_text_ds_job.insert_rate * num_not_inserted
                )

                without_inserted = " ".join(
                    [w for w, t in zip(example["words"], example["target"]) if t == 0]
                )

                assert without_inserted in texts
