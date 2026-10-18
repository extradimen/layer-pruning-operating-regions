"""Non-evidential integrity and disclosure checks for the release candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def count_csv(path: str) -> int:
    with (HERE / path).open(newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def read_csv(path: str) -> list[dict]:
    with (HERE / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    lines = (HERE / "MANIFEST.sha256").read_text().splitlines()
    listed = {line.split("  ", 1)[1] for line in lines}
    actual_files = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts and "rebuilt_figures" not in p.parts and p.suffix != ".pyc"
                    and p.name not in {"MANIFEST.sha256", "SOURCE_MAP.csv", "build_release.py"}}
    assert listed == actual_files, (sorted(listed - actual_files), sorted(actual_files - listed))
    for line in lines:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
    expected_counts = {
        "02_selection/results/logical_endpoints.csv": 96,
        "02_selection/results/dense_shared_endpoints.csv": 2,
        "04_evaluation/results/question_metrics.csv": 23232,
        "04_evaluation/results/paired_serving_trials.csv": 6400,
        "05_statistics/results/A_configurations.csv": 52,
        "05_statistics/results/B_configurations.csv": 196,
        "05_statistics/results/A_cost_events.csv": 7874,
        "05_statistics/results/B_cost_events.csv": 37853,
        "07_boundary/results/historical_logical_points.csv": 24,
        "07_boundary/results/historical_question_metrics.csv": 8096,
        "07_boundary/results/qwen_length_question_metrics.csv": 192,
        "07_boundary/results/mistral_paired_serving_trials.csv": 1000,
    }
    for path, expected in expected_counts.items():
        assert count_csv(path) == expected, path
    endpoints = read_csv("02_selection/results/logical_endpoints.csv")
    expected_cells = {(model, method, str(seed), str(k))
                      for model in ("qwen", "mistral")
                      for method in ("interaction_tabu", "shortgpt_bi_adapted", "sleb_iterative_adapted", "random")
                      for seed in (11, 23, 37) for k in (1, 2, 3, 4)}
    assert {(r["model"], r["method"], r["seed"], r["k"]) for r in endpoints} == expected_cells
    physical = {(r["model"], r["topology_id"]) for r in endpoints}
    questions = read_csv("04_evaluation/results/question_metrics.csv")
    timings = read_csv("04_evaluation/results/paired_serving_trials.csv")
    assert physical == {(r["model"], r["topology_id"]) for r in questions if r["topology_id"] != "DENSE"}
    assert physical == {(r["model"], r["topology_id"]) for r in timings}
    assert {(r["model"], r["topology_id"]) for r in questions if r["topology_id"] == "DENSE"} == {("qwen", "DENSE"), ("mistral", "DENSE")}
    for phase, expected in (("A", 18), ("B", 96)):
        data = json.loads((HERE / f"05_statistics/results/{phase}_statistics.json").read_text())
        assert len(data["contrasts"]) == expected
    forbidden = re.compile(rb"/Users/|/root/codex-projects/|10\.63\.0\.|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:hf_|sk-)[A-Za-z0-9]{20,}")
    for line in lines:
        _, relative = line.split("  ", 1)
        if relative in {"verify_release.py", "RELEASE_GATES.md", "README.md"}:
            continue
        assert not forbidden.search((HERE / relative).read_bytes()), relative
    print(f"PASS: {len(lines)} frozen files, study counts, and disclosure scan")


if __name__ == "__main__":
    main()
