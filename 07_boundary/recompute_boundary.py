"""Recompute historical Holm-9 families and length summaries from released rows."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE / "05_statistics"))
from recompute_primary import compare, holm, test_statistic  # noqa: E402

TASKS = ("piqa", "hellaswag", "gsm8k")
ARMS = ("k3", "k6", "k9", "branch")
DIRECTIONS = ("k6-k3", "k9-k6", "branch-k6block")


def rows(path: str) -> list[dict]:
    with (HERE / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def quality_check(model: str, seed: int) -> dict:
    points = {(r["arm"], int(r["seed"])): r["topology_id"]
              for r in rows("07_boundary/results/historical_logical_points.csv") if r["model"] == model}
    assert set(points) == {(arm, s) for arm in ARMS for s in (11, 23, 37)}
    outcomes = defaultdict(dict)
    for row in rows("07_boundary/results/historical_question_metrics.csv"):
        if row["model"] == model and row["task"] in TASKS:
            key = row["topology_id"], row["task"]
            assert row["sample_id"] not in outcomes[key]
            outcomes[key][row["sample_id"]] = float(row["correct"])
    frozen = json.loads((HERE / f"07_boundary/results/{model}_quality.json").read_text())
    contrasts, per_seed = [], []
    raw_p = [1.0] * 9
    for task_i, task in enumerate(TASKS):
        baseline = outcomes["DENSE", task]
        ids = sorted(baseline)
        assert len(ids) == (32 if task == "gsm8k" else 128)
        ix = np.random.default_rng(seed).integers(0, len(ids), size=(50000, len(ids)))
        arms = {}
        for arm in ARMS:
            vectors = []
            for fixed_seed in (11, 23, 37):
                topology = points[arm, fixed_seed]
                sample = outcomes[topology, task]
                assert set(sample) == set(ids)
                vector = np.array([sample[i] for i in ids], dtype=float)
                vectors.append(vector)
                per_seed.append({"arm": arm, "seed": fixed_seed, "task": task,
                                 "artifact": topology, "accuracy": float(vector.mean()),
                                 "correct": int(vector.sum()), "n": len(vector)})
            arms[arm] = np.asarray(vectors)
        for direction_i, (left, right) in enumerate((("k6", "k3"), ("k9", "k6"), ("branch", "k6"))):
            diff = arms[left].mean(axis=0) - arms[right].mean(axis=0)
            observed, interval, p = test_statistic(diff, seed)
            slot = direction_i * 3 + task_i
            raw_p[slot] = p
            contrasts.append({"task": task, "direction": DIRECTIONS[direction_i],
                              "family_slot": slot, "accuracy_difference": observed,
                              "nominal95CI": interval, "p_two_sided": p,
                              "question_clusters": len(ids)})
        for arm in ARMS:
            mean = arms[arm].mean(axis=0)
            expected = frozen["quality"][task]["arms"][arm]
            compare(float(mean.mean()), expected["accuracy"])
            compare(np.quantile(mean[ix].mean(axis=1), [.025, .975]).tolist(), expected["nominal95CI"])
        compare(float(np.mean(list(baseline.values()))), frozen["quality"][task]["dense_accuracy"])
    adjusted = holm(raw_p)
    for contrast in contrasts:
        p = adjusted[contrast["family_slot"]]
        contrast.update(p_Holm9=p, reject_Holm9=p < .05)
    compare(per_seed, frozen["per_seed_quality"])
    checks = compare(contrasts, frozen["contrasts"])
    return {"model": model, "contrasts": len(contrasts), "checked_values": checks}


def length_check() -> dict:
    data = rows("07_boundary/results/qwen_length_question_metrics.csv")
    assert len(data) == 192
    groups = defaultdict(list)
    for row in data:
        groups[row["arm"], int(row["max_new_tokens"])].append(row)
    assert len(groups) == 12 and all(len(group) == 16 for group in groups.values())
    frozen = json.loads((HERE / "07_boundary/results/qwen_length.json").read_text())
    checked = 0
    for summary in frozen["summaries"]:
        group = groups[summary["config"], summary["length"]]
        actual = {"n": len(group), "correct": sum(int(row["correct"]) for row in group),
                  "truncated": sum(row["truncated"] == "True" for row in group),
                  "parse_failure": sum(row["parse_failure"] == "True" for row in group)}
        for key, value in actual.items():
            checked += compare(value, summary[key])
        checked += compare(sum(float(row["wall_seconds"]) for row in group), summary["wall_seconds"]["total"])
        checked += compare(sum(int(row["output_tokens"]) for row in group), summary["output_tokens"]["total"])
    return {"length_cells": len(groups), "checked_values": checked}


def serving_check() -> dict:
    data = rows("07_boundary/results/mistral_paired_serving_trials.csv")
    assert len(data) == 1000
    groups = defaultdict(list)
    for row in data:
        groups[row["topology_id"], row["workload"]].append(row)
    assert len(groups) == 20
    rng = np.random.default_rng(20261025)
    calculated = []
    for topology, workload in sorted(groups):
        measurements = groups[topology, workload]
        assert len(measurements) == 50
        dense = {int(r["trial"]): r for r in measurements if r["warmup"] == "False" and r["arm"] == "dense"}
        candidate = {int(r["trial"]): r for r in measurements if r["warmup"] == "False" and r["arm"] == "candidate"}
        assert set(dense) == set(candidate) == set(range(5, 25))
        assert all(dense[t]["pair_order"] == candidate[t]["pair_order"] for t in dense)
        dv = np.array([float(dense[t]["wall_seconds"]) for t in sorted(dense)])
        cv = np.array([float(candidate[t]["wall_seconds"]) for t in sorted(candidate)])
        ix = rng.integers(0, 20, size=(50000, 20))
        ratio = dv[ix].mean(axis=1) / cv[ix].mean(axis=1)
        delta = (cv - dv)[ix].mean(axis=1)
        calculated.append({"topology_id": topology, "workload": workload, "n_pairs": 20,
                           "candidate_mean_seconds": float(cv.mean()),
                           "dense_mean_seconds": float(dv.mean()),
                           "speedup_ratio": float(dv.mean() / cv.mean()),
                           "speedup_nominal95_low": float(np.quantile(ratio, .025)),
                           "speedup_nominal95_high": float(np.quantile(ratio, .975)),
                           "paired_delta_seconds": float((cv - dv).mean()),
                           "delta_nominal95_low": float(np.quantile(delta, .025)),
                           "delta_nominal95_high": float(np.quantile(delta, .975)),
                           "candidate_peak_allocated_bytes": max(int(r["peak_allocated_bytes"]) for r in candidate.values()),
                           "dense_peak_allocated_bytes": max(int(r["peak_allocated_bytes"]) for r in dense.values())})
    frozen = json.loads((HERE / "07_boundary/results/mistral_serving.json").read_text())
    return {"serving_cells": len(calculated), "checked_values": compare(calculated, frozen["cells"])}


def main() -> None:
    result = {"status": "PASS_RELEASED_BOUNDARY_ROWS_MATCH_FROZEN",
              "quality": [quality_check("qwen", 20261021), quality_check("mistral", 20261024)],
              "length": length_check(), "mistral_serving": serving_check()}
    output = HERE / "07_boundary/results/recompute_boundary_check.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
