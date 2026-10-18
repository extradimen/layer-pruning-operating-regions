"""Recompute the frozen A/B accuracy families and paired serving summaries.

Only curated released rows are read. No model inference or source-project paths.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
METHODS = ("interaction_tabu", "shortgpt_bi_adapted", "sleb_iterative_adapted", "random")
MODELS = ("qwen", "mistral")
TASKS = ("piqa", "hellaswag", "gsm8k")
SEEDS = (11, 23, 37)


def load_csv(relative: str) -> list[dict]:
    with (HERE / relative).open(newline="") as stream:
        return list(csv.DictReader(stream))


def holm(raw: list[float]) -> list[float]:
    n = len(raw)
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(sorted(range(n), key=lambda i: (raw[i], i))):
        running = max(running, min(1.0, (n - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def make_rows() -> tuple[dict, dict, dict]:
    endpoints = {(r["model"], r["method"], int(r["seed"]), int(r["k"])): r["topology_id"]
                 for r in load_csv("02_selection/results/logical_endpoints.csv")}
    assert len(endpoints) == 96
    outcomes = defaultdict(dict)
    for row in load_csv("04_evaluation/results/question_metrics.csv"):
        if row["task"] not in TASKS:
            continue
        key = (row["model"], row["topology_id"], row["task"])
        assert row["sample_id"] not in outcomes[key], key
        outcomes[key][row["sample_id"]] = float(row["correct"])
    return endpoints, outcomes, load_csv("04_evaluation/results/paired_serving_trials.csv")


def vectors(model: str, task: str, endpoints: dict, outcomes: dict) -> tuple[list[str], np.ndarray, dict]:
    dense = outcomes[model, "DENSE", task]
    ids = sorted(dense)
    assert len(ids) == (32 if task == "gsm8k" else 128)
    baseline = np.array([dense[i] for i in ids], dtype=float)
    arms = {}
    for k in (1, 2, 3, 4):
        for method in METHODS:
            values = []
            for seed in SEEDS:
                actual = outcomes[model, endpoints[model, method, seed, k], task]
                assert set(actual) == set(ids), (model, method, seed, k, task)
                values.append([actual[i] for i in ids])
            arms[method, k] = np.asarray(values, dtype=float)
    return ids, baseline, arms


def test_statistic(diff: np.ndarray, seed: int) -> tuple[float, list[float], float]:
    n = len(diff)
    bootstrap = np.random.default_rng(seed).integers(0, n, size=(50000, n))
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(50000, n))
    observed = float(diff.mean())
    interval = np.quantile(diff[bootstrap].mean(axis=1), [.025, .975]).tolist()
    null = (signs * diff).mean(axis=1)
    p = (int((np.abs(null) >= abs(observed) - 1e-12).sum()) + 1) / 50001
    return observed, interval, p


def make_contrasts(endpoints: dict, outcomes: dict) -> tuple[list[dict], list[dict]]:
    phase_a, phase_b = [], []
    cache = {(model, task): vectors(model, task, endpoints, outcomes)
             for model in MODELS for task in TASKS}
    for model in MODELS:
        for task in TASKS:
            ids, _, arms = cache[model, task]
            for method in METHODS[1:]:
                diff = (arms[METHODS[0], 3] - arms[method, 3]).mean(axis=0)
                observed, interval, p = test_statistic(diff, 20261026)
                phase_a.append({"model": model, "task": task,
                                "comparison": "interaction_tabu-minus-" + method,
                                "difference": observed, "nominal95CI": interval,
                                "p_two_sided": p, "question_clusters": len(ids)})
    for row, adjusted in zip(phase_a, holm([r["p_two_sided"] for r in phase_a])):
        row.update(p_Holm18=adjusted, reject_Holm18=adjusted < .05)
    for model in MODELS:
        for k in (1, 2, 3, 4):
            for method in METHODS:
                for task in TASKS:
                    ids, dense, arms = cache[model, task]
                    diff = (arms[method, k] - dense).mean(axis=0)
                    observed, interval, p = test_statistic(diff, 20261027)
                    phase_b.append({"model": model, "k": k, "method": method,
                                    "task": task, "comparison": method + "-minus-Dense",
                                    "accuracy_difference": observed, "nominal95CI": interval,
                                    "p_two_sided": p, "question_clusters": len(ids)})
    for row, adjusted in zip(phase_b, holm([r["p_two_sided"] for r in phase_b])):
        row.update(p_Holm96=adjusted, reject_Holm96=adjusted < .05)
    return phase_a, phase_b


def make_serving(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        if row["warmup"] == "False":
            groups[row["topology_id"], row["workload"]].append(row)
    serving = defaultdict(dict)
    for (topology, workload), observations in groups.items():
        dense = {int(row["trial"]): row for row in observations if row["arm"] == "dense"}
        candidate = {int(row["trial"]): row for row in observations if row["arm"] == "candidate"}
        assert set(dense) == set(candidate) == set(range(5, 25))
        assert all(dense[t]["pair_order"] == candidate[t]["pair_order"] for t in dense)
        dv = np.array([float(dense[t]["wall_seconds"]) for t in sorted(dense)])
        cv = np.array([float(candidate[t]["wall_seconds"]) for t in sorted(candidate)])
        indices = np.random.default_rng(20261027).integers(0, 20, size=(50000, 20))
        ratios = dv[indices].mean(axis=1) / cv[indices].mean(axis=1)
        serving[topology][workload] = {
            "dense_mean_s": float(dv.mean()), "candidate_mean_s": float(cv.mean()),
            "speedup": float(dv.mean() / cv.mean()),
            "speedup_CI_nominal95": np.quantile(ratios, [.025, .975]).tolist(),
            "peak_allocated_bytes": max(int(row["peak_allocated_bytes"]) for row in candidate.values()),
            "pairs": 20,
        }
    return dict(serving)


def compare(actual: object, expected: object, path: str = "root") -> int:
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected), path
        return sum(compare(actual[key], expected[key], path + "." + key) for key in expected)
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        return sum(compare(a, e, path + f"[{i}]") for i, (a, e) in enumerate(zip(actual, expected)))
    if isinstance(expected, bool) or isinstance(expected, str) or isinstance(expected, int):
        assert actual == expected, (path, actual, expected)
    else:
        assert abs(float(actual) - float(expected)) <= 1e-12, (path, actual, expected)
    return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    endpoints, outcomes, paired_rows = make_rows()
    phase_a, phase_b = make_contrasts(endpoints, outcomes)
    serving = make_serving(paired_rows)
    frozen_a = json.loads((HERE / "05_statistics/results/A_statistics.json").read_text())
    frozen_b = json.loads((HERE / "05_statistics/results/B_statistics.json").read_text())
    checks = {"A_contrasts": compare(phase_a, frozen_a["contrasts"]),
              "B_contrasts": compare(phase_b, frozen_b["contrasts"]),
              "B_serving": compare(serving, frozen_b["serving"])}
    result = {"status": "PASS_RELEASED_ROWS_MATCH_FROZEN_PRIMARY", "checks": checks,
              "A_contrasts": len(phase_a), "B_contrasts": len(phase_b),
              "physical_serving_artifacts": len(serving), "question_cluster_unit": "sample_id"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
