#!/usr/bin/env python3
import json
import sys

from predwordimp.data.dataset_job import WikiTextDsJob
from predwordimp.eval.wi_eval import EvalWordImp
from predwordimp.training.train import TrainJob
from predwordimp.util.job import ConfigurableJob

job_classes: dict[str, type[ConfigurableJob]] = {
    "WikiTextDsJob": WikiTextDsJob,
    "TrainJob": TrainJob,
    "EvalWordImp": EvalWordImp,
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run.py <config_json_file>")
        sys.exit(1)

    json_file = sys.argv[1]
    with open(file=json_file, mode="r") as file:
        data = json.load(file)

    job_type = data.pop("job", None)

    job = job_classes[job_type].from_dict(data=data)
    job.run()
