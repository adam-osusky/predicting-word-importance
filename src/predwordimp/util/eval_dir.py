import json
import os
import subprocess
import sys

JSON_NAME = "eval.json"
MAX_RANK_LIMIT = 0.1


def generate_json_file(experiment_dir: str, experiment_name: str) -> None:
    data = {
        "job": "EvalWordImp",
        "hf_model": os.path.join(experiment_dir, experiment_name, experiment_name),
        "max_rank_limit": MAX_RANK_LIMIT,
        "seed": 69,
    }

    with open(JSON_NAME, "w") as json_file:
        json.dump(data, json_file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python eval_dir.py <dir_with_models>")
        sys.exit(1)

    directory_path = sys.argv[1]
    directories = os.listdir(directory_path)
    for directory in directories:
        generate_json_file(directory_path, directory)
        subprocess.run(["python", "src/predwordimp/util/run.py", JSON_NAME])

    os.remove(JSON_NAME)
