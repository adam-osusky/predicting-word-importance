import json
import os
import subprocess
import sys

JSON_NAME = "eval.json"
MAX_RANK_LIMIT = 0.1


def generate_json_file(experiment_dir: str, experiment_name: str, domain: str) -> None:
    data = {
        "job": "EvalWordImp",
        "hf_model": os.path.join(experiment_dir, experiment_name, experiment_name),
        "max_rank_limit": MAX_RANK_LIMIT,
        "domain": domain,
        "seed": 69,
    }

    with open(JSON_NAME, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    """
    Main entry point of the script. Iterates over directory containing directories
    for experiments. One experiment have one dir. Runs the evaluation job on 
    Word Importance Dataset (WIDS).

    Usage:
        python eval_dir.py <dir_with_models> [domain]

    Args:
        dir_with_models (str): Path to the directory containing experiment directories.
        domain (str, optional): The domain of the WIDS to eval on. Default is an empty string.

    Example:
        python eval_dir.py models/ text-domain
    """
    if len(sys.argv) not in [2, 3]:
        print("Usage: python eval_dir.py <dir_with_models> [domain]")
        sys.exit(1)

    domain = ""
    if len(sys.argv) == 3:
        domain = sys.argv[2]

    directory_path = sys.argv[1]
    directories = os.listdir(directory_path)
    for directory in directories:
        generate_json_file(directory_path, directory, domain)
        subprocess.run(["python", "src/predwordimp/util/run.py", JSON_NAME])

    os.remove(JSON_NAME)
