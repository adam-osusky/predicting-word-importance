import json
import math
import os


def get_rank_limit(limit: int | float, length: int) -> int:
    if isinstance(limit, int):
        return limit
    elif isinstance(limit, float):
        return math.ceil(length * limit)


def get_model_name(model_path: str, ds_dir: str = "data/metacentrum/wikitext") -> str:
    if "adasgaleus" in model_path:
        return model_path.split("/")[1]

    experiment_dir = os.path.dirname(model_path)

    with open(os.path.join(experiment_dir, "train_config.json")) as f:
        train_config = json.load(f)
        dataset_name = os.path.basename(os.path.normpath(train_config["dataset_dir"]))

    with open(os.path.join(ds_dir, dataset_name, "config.json")) as f:
        ds_config = json.load(f)

    model = "bert" if "bert" in ds_config["insert_model"] else ds_config["insert_model"]

    return f"{model}-{ds_config['insert_rate']}"
