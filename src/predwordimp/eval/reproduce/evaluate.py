import json
import os
import random
import subprocess
from typing import Any

import numpy as np

from predwordimp.eval.metrics import RankingEvaluator
from predwordimp.eval.wi_eval import EvalWordImp, get_rank_limit, load_ds, rankdata

JSON_NAME = "eval.json"


def generate_json_file(model_name: str) -> None:
    data = {
        "job": "EvalWordImp",
        "hf_model": model_name,
        "max_rank_limit": 0.1,
        "domain": "",
        "seed": 69,
    }

    with open(JSON_NAME, "w") as json_file:
        json.dump(data, json_file)


def generate_tfidf_json_file() -> None:
    data = {
        "job": "TF-IDF",
        "max_rank_limit": 0.1,
        "domain": "",
        "seed": 69,
    }

    with open(JSON_NAME, "w") as json_file:
        json.dump(data, json_file)


config = {
    "job_name": "",
    "hf_model": "",
    "max_rank_limit": 0.1,
    "seed": 69,
}

eval_job = EvalWordImp(**config)


def sorted2rank(x: Any, method: Any) -> Any:
    limit = get_rank_limit(eval_job.max_rank_limit, len(x))
    rank = np.ones(len(x)) * (limit + 1)

    if method == "normal":
        for i, idx in enumerate(x[:limit]):
            rank[idx] = i + 1

    elif method == "random":
        for i, idx in enumerate(random.sample(range(1, len(x)), limit)):
            rank[idx] = i + 1

    elif method == "norank":
        pass

    return rank


def ignore_tokens_scores(scores: Any, ds: Any) -> Any:
    new_scores = []

    for i in range(len(scores)):
        new_score = scores[i]
        words = ds[i]["context"]
        for j, w in enumerate(words):
            if w.startswith("(PERSON"):
                new_score[j] = 0.0
        new_scores.append(new_score)

    return new_scores


def get_results(json_preds: Any, method: str) -> dict[Any, Any]:
    """Get evaluation results from a predicted scores in a json."""

    scores = [sentence["scores"] for sentence in json_preds["sentences"]]

    sorted_scores = [np.argsort(score)[::-1] for score in scores]

    ranks = [sorted2rank(sorted_score, method) for sorted_score in sorted_scores]

    ranks = RankingEvaluator.ignore_maximal(ranks, eval_job.max_rank_limit)

    ds = load_ds()
    labels = ds["label"]

    labels = [rankdata(label) for label in labels]
    labels = RankingEvaluator.ignore_maximal(labels, rank_limit=eval_job.max_rank_limit)

    results = {}
    results["name"] = json_preds["name"]

    results["pearson"], p_vals = RankingEvaluator.mean_rank_correlation(
        ranks, labels, "pearson"
    )
    results["pearson_pvalue"] = p_vals[0]

    results["kendall"], p_vals = RankingEvaluator.mean_rank_correlation(
        ranks, labels, "kendall"
    )
    results["kendall_pvalue"] = p_vals[0]

    results["somers"] = RankingEvaluator.mean_rank_correlation(ranks, labels, "somers")

    for k in range(1, 6):
        k_inter = f"{k}-inter"
        results[k_inter] = RankingEvaluator.least_intersection(ranks, labels, k)

    results["avg_overlap"] = RankingEvaluator.avg_overlaps(
        ranks, labels, limit=eval_job.max_rank_limit
    )

    return results


if __name__ == "__main__":
    model_names = [
        "adasgaleus/LIM-0.25",
        "adasgaleus/LIM-0.5",
        "adasgaleus/LIM-0.75",
        "adasgaleus/BIM-0.25",
        "adasgaleus/BIM-0.5",
        "adasgaleus/BIM-0.75",
    ]

    for model in model_names:
        generate_json_file(model)
        subprocess.run(["python", "src/predwordimp/util/run.py", JSON_NAME])

    generate_tfidf_json_file()
    subprocess.run(["python", "src/predwordimp/util/run.py", JSON_NAME])

    results = []
    eval_dir = "./data/eval"
    for dir in os.listdir(eval_dir):
        with open(os.path.join(eval_dir, dir, "results.json"), "r") as f:
            results.append(json.load(f))

    with open("src/predwordimp/eval/reproduce/WI-NLI.json", "r") as f:
        nli_preds = json.load(f)
        nli_preds["name"] = "NLI"
    with open("src/predwordimp/eval/reproduce/WI-PI.json", "r") as f:
        pi_preds = json.load(f)
        pi_preds["name"] = "PI"
    results.append(get_results(nli_preds, method="normal"))
    results.append(get_results(pi_preds, method="normal"))
    pi_preds["name"] = "random-ranking"
    results.append(get_results(pi_preds, method="random"))

    with open("results.json", "w") as file:
        json.dump(results, file, indent=4)

    os.remove(JSON_NAME)
