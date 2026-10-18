"""Build the curated public release from frozen local evidence (no experiments)."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "outputs/twelfth-selection-v12"
SOURCES: list[tuple[str, str, str]] = []


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def register(source: Path, destination: Path, transform: str) -> None:
    SOURCES.append((str(source.relative_to(ROOT)), str(destination.relative_to(OUT)), digest(source) + ";" + transform))


def copy(source: str, destination: str) -> None:
    src, dst = ROOT / source, OUT / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    register(src, dst, "byte-identical")


def write_csv(destination: str, rows: list[dict]) -> None:
    dst = OUT / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(destination + " has no rows")
    with dst.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(source: str) -> dict:
    return json.loads((ROOT / source).read_text())


def main() -> None:
    protocol_files = {
        "01_design/phase_A_protocol.zh.txt": "outputs/twelfth-selection-v12/protocol-v1/PROTOCOL.zh.txt",
        "01_design/phase_A_design.json": "outputs/twelfth-selection-v12/protocol-v1/design_v1.json",
        "01_design/phase_B_protocol.zh.txt": "outputs/twelfth-selection-v12/phase_B_protocol_v1/PROTOCOL.zh.txt",
        "01_design/phase_B_design.json": "outputs/twelfth-selection-v12/phase_B_protocol_v1/design_v1.json",
        "01_design/source_mapping.csv": "outputs/twelfth-selection-v12/protocol-v1/source_mapping.csv",
    }
    for dst, src in protocol_files.items():
        copy(src, dst)

    stages = {
        "02_selection/A": ["stage-F-masks-v1", "stage-J-masks-prepared-v1", "stage-K-selection-pilot-v2", "stage-K-selection-prepared-v1"],
        "02_selection/B": ["stage-L-masks-prepared-v1", "stage-L-selection-prepared-v1"],
        "03_artifacts/A": ["stage-K-artifacts-prepared-v1"],
        "03_artifacts/B": ["stage-L-artifacts-prepared-v2"],
        "04_evaluation/A": ["stage-K-quality-prepared-v1", "stage-K-paired-prepared-v2"],
        "04_evaluation/B": ["stage-L-quality-prepared-v1", "stage-L-paired-prepared-v2"],
        "05_statistics/A": ["stage-K-statistics-v1"],
        "05_statistics/B": ["stage-L-statistics-v1"],
        "06_figures/A": ["stage-K-evidence-v1"],
        "06_figures/B": ["stage-L-evidence-v1"],
        "07_boundary/code": ["stage-J-statistics-v1", "stage-I-length-statistics-v1"],
    }
    for section, directories in stages.items():
        for directory in directories:
            for source in sorted((ROOT / "work" / directory).glob("*.py")):
                copy(str(source.relative_to(ROOT)), f"{section}/{directory}/{source.name}")
            for source in sorted((ROOT / "work" / directory).glob("*.schema.json")):
                copy(str(source.relative_to(ROOT)), f"{section}/{directory}/{source.name}")
    copy("manuscript/neurocomputing/revised_full_manuscript_v2/build_readable_figures.py", "06_figures/paper/build_readable_figures.py")
    original_figure_source = ROOT / "manuscript/neurocomputing/revised_full_manuscript_v2/build_readable_figures.py"
    figure_code = original_figure_source.read_text()
    figure_code = figure_code.replace("ROOT = Path(__file__).resolve().parents[3]", "ROOT = Path(__file__).resolve().parents[2]")
    figure_code = figure_code.replace("OUT = Path(__file__).resolve().parent / 'figures'", "OUT = Path(__file__).resolve().parent / 'rebuilt_figures'\nOUT.mkdir(exist_ok=True)")
    figure_code = figure_code.replace("ROOT/'outputs/twelfth-selection-v12/phase_B_operational_v1/statistics-return-v1/statistics.json'", "ROOT/'05_statistics/results/B_statistics.json'")
    figure_code = figure_code.replace("ROOT/'outputs/twelfth-selection-v12/statistics-return-v2/statistics.json'", "ROOT/'05_statistics/results/A_statistics.json'")
    portable_figure = OUT / "06_figures/paper/build_paper_figures.py"
    portable_figure.write_text(figure_code)
    register(original_figure_source, portable_figure, "portable-input-path-adaptation")

    summary_sources = {
        "A": "outputs/twelfth-selection-v12/statistics-return-v2",
        "B": "outputs/twelfth-selection-v12/phase_B_operational_v1/statistics-return-v1",
    }
    for phase, source in summary_sources.items():
        copy(source + "/all_configurations_long.csv", f"05_statistics/results/{phase}_configurations.csv")
        statistics = read_json(source + "/statistics.json")
        public_stats = {name: statistics[name] for name in ("status", "contrasts", "long_table", "serving", "dense_baseline", "family", "replicates", "seed", "scope", "cost_boundary", "cost_lease_lineage", "shared_physical_not_independent", "cost_event_count") if name in statistics}
        dst = OUT / f"05_statistics/results/{phase}_statistics.json"
        dst.write_text(json.dumps(public_stats, ensure_ascii=False, indent=2) + "\n")
        register(ROOT / (source + "/statistics.json"), dst, "selected-public-fields")
        with (ROOT / (source + "/cost_events_long.csv")).open(newline="") as stream:
            events = list(csv.DictReader(stream))
        for row in events:
            row.pop("source_archive", None)
        write_csv(f"05_statistics/results/{phase}_cost_events.csv", events)
        register(ROOT / (source + "/cost_events_long.csv"), OUT / f"05_statistics/results/{phase}_cost_events.csv", "removed-source_archive")

    boundary_sources = {
        "mistral_quality": "outputs/eleventh-transfer-v11/statistics-v1/statistics.json",
        "mistral_serving": "outputs/eleventh-transfer-v11/serving_statistics-v1/statistics.json",
        "qwen_quality": "outputs/tenth-confirmation-v10/report-v1/frozen_E1_statistics.json",
        "qwen_length": "outputs/tenth-confirmation-v10/report-v1/frozen_length_statistics.json",
    }
    for label, source in boundary_sources.items():
        data = read_json(source)
        data.pop("inputs", None)
        dst = OUT / f"07_boundary/results/{label}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        register(ROOT / source, dst, "removed-input-file-paths")

    historical_quality_sources = {
        "qwen": "outputs/tenth-confirmation-v10/quality_report.json",
        "mistral": "outputs/eleventh-transfer-v11/quality-return-v1/stage-J-quality-v1.json",
    }
    historical_rows = []
    historical_points = []
    for model, source in historical_quality_sources.items():
        report = read_json(source)
        points = report["logical_points"]
        if isinstance(points, dict):
            historical_points.extend({"model": model, "arm": ("branch" if "branches" in label else label.split("-")[1]), "seed": int(label.rsplit("s", 1)[1]),
                                      "topology_id": topology} for label, topology in points.items())
        else:
            historical_points.extend({"model": model, "arm": point["arm"], "seed": point["seed"],
                                      "topology_id": point["topology_id"]} for point in points)
        bundles = {"DENSE": report["dense"]}
        bundles.update({topology: artifact["quality"] for topology, artifact in report["artifacts"].items()})
        for topology, tasks in bundles.items():
            for task, rows in tasks.items():
                for index, row in enumerate(rows):
                    historical_rows.append({
                        "model": model, "topology_id": topology, "task": task, "row_index": index,
                        "sample_id": row.get("id", ""), "correct": row.get("correct", ""),
                        "nll": row.get("nll", ""), "effective_loss_tokens": row.get("effective_loss_tokens", ""),
                        "parse_failure": row.get("parse_failure", ""), "truncated": row.get("truncated", ""),
                        "input_tokens": row.get("input_tokens", ""), "output_tokens": row.get("output_tokens", ""),
                        "wall_seconds": row.get("wall_seconds", ""),
                    })
        register(ROOT / source, OUT / "07_boundary/results/historical_question_metrics.csv", "selected-metrics-no-text-or-labels")
        register(ROOT / source, OUT / "07_boundary/results/historical_logical_points.csv", "selected-point-identities")
    write_csv("07_boundary/results/historical_question_metrics.csv", historical_rows)
    write_csv("07_boundary/results/historical_logical_points.csv", historical_points)

    length_source = "outputs/tenth-confirmation-v10/report-v1/frozen_length_statistics.json"
    length_report = read_json(length_source)
    length_rows = []
    for row in length_report["raw_pairs"]:
        length_rows.append({"arm": row["config"], "sample_id": row["id"],
                            "max_new_tokens": row["length"], "correct": row["correct"],
                            "parse_failure": row["parse_failure"],
                            "truncated": row["truncated"],
                            "output_tokens": row["output_tokens"],
                            "wall_seconds": row["wall_seconds"]})
    write_csv("07_boundary/results/qwen_length_question_metrics.csv", length_rows)
    register(ROOT / length_source, OUT / "07_boundary/results/qwen_length_question_metrics.csv", "selected-length-metrics-no-generated-text-or-labels")

    older_paired_source = "outputs/eleventh-transfer-v11/paired-resume-return-v1/stage-J-paired-resume-v1.json"
    older_paired = read_json(older_paired_source)
    older_paired_rows = []
    for topology, artifact in sorted(older_paired["artifacts"].items()):
        for workload, observations in artifact["measurements"].items():
            for observation in observations:
                older_paired_rows.append({"topology_id": topology, "workload": workload,
                                         "trial": observation["trial"], "warmup": observation["warmup"],
                                         "arm": observation["model"],
                                         "pair_order": "/".join(observation["pair_order"]),
                                         "generated_tokens": observation["generated_tokens"],
                                         "wall_seconds": observation["wall_seconds"],
                                         "peak_allocated_bytes": observation["peak_allocated_bytes"]})
    write_csv("07_boundary/results/mistral_paired_serving_trials.csv", older_paired_rows)
    register(ROOT / older_paired_source, OUT / "07_boundary/results/mistral_paired_serving_trials.csv", "selected-timing-and-memory-fields")

    for model, source in {
        "qwen": "outputs/seventh-exploration-v7/mask_report.json",
        "mistral": "outputs/eleventh-transfer-v11/mask_report_v1.json",
    }.items():
        report = read_json(source)
        records = report.get("results", report.get("records", []))
        rows = [{"model": model, "seed": row["seed"], "k": row.get("k", 3),
                 "mask": json.dumps(row["mask"]), "solver": json.dumps(row["solver"], separators=(",", ":"))}
                for row in records if row.get("k", 3) == 3 and ("arm" not in row or row["arm"] == "k3")]
        write_csv(f"02_selection/results/{model}_historical_tabu_k3.csv", rows)
        register(ROOT / source, OUT / f"02_selection/results/{model}_historical_tabu_k3.csv", "selected-k3-mask-and-solver-fields")

    quality_files = ["outputs/twelfth-selection-v12/quality-return-v1/results/stage-K-quality-v1.json"]
    quality_files += [str(p.relative_to(ROOT)) for p in sorted((EVIDENCE / "phase_B_operational_v1/quality-return-v1").glob("*.json"))]
    quality_rows = []
    logical_rows = []
    dense_rows = []
    seen_quality = set()
    seen_logical = set()
    for source in quality_files:
        data = read_json(source)
        for point in data["logical_points"]:
            key = (point["model"], point["method"], point["seed"], point["k"])
            if key not in seen_logical:
                logical_rows.append({name: json.dumps(point[name]) if name == "mask" else point.get(name, "") for name in ("model", "method", "seed", "k", "mask", "topology_id", "bundle_SHA256")})
                seen_logical.add(key)
        for model, model_data in data["models"].items():
            if "dense" in model_data:
                dense_rows.append({"model": model, "method": "dense_shared", "k": 0, "topology_id": "DENSE"})
                for task, rows in model_data["dense"].items():
                    for index, row in enumerate(rows):
                        quality_rows.append({
                            "model": model, "topology_id": "DENSE", "task": task,
                            "row_index": index, "sample_id": row.get("id", ""),
                            "correct": row.get("correct", ""), "nll": row.get("nll", ""),
                            "effective_loss_tokens": row.get("effective_loss_tokens", ""),
                            "parse_failure": row.get("parse_failure", ""), "truncated": row.get("truncated", ""),
                            "input_tokens": row.get("input_tokens", ""), "output_tokens": row.get("output_tokens", ""),
                            "actual_forward_calls": row.get("actual_forward_calls", ""), "wall_seconds": row.get("wall_seconds", ""),
                        })
            for topology_id, artifact in model_data["artifacts"].items():
                for task, rows in artifact["quality"].items():
                    for index, row in enumerate(rows):
                        key = (model, topology_id, task, index)
                        if key in seen_quality:
                            continue
                        quality_rows.append({
                            "model": model, "topology_id": topology_id, "task": task,
                            "row_index": index, "sample_id": row.get("id", ""),
                            "correct": row.get("correct", ""), "nll": row.get("nll", ""),
                            "effective_loss_tokens": row.get("effective_loss_tokens", ""),
                            "parse_failure": row.get("parse_failure", ""), "truncated": row.get("truncated", ""),
                            "input_tokens": row.get("input_tokens", ""), "output_tokens": row.get("output_tokens", ""),
                            "actual_forward_calls": row.get("actual_forward_calls", ""), "wall_seconds": row.get("wall_seconds", ""),
                        })
                        seen_quality.add(key)
        register(ROOT / source, OUT / "04_evaluation/results/question_metrics.csv", "selected-fields;no-inputs-labels-or-generated-text")
        register(ROOT / source, OUT / "02_selection/results/logical_endpoints.csv", "selected-logical-point-fields")
    write_csv("02_selection/results/logical_endpoints.csv", sorted(logical_rows, key=lambda r: (r["model"], int(r["k"]), r["method"], int(r["seed"]))))
    write_csv("02_selection/results/dense_shared_endpoints.csv", dense_rows)
    write_csv("04_evaluation/results/question_metrics.csv", quality_rows)

    paired_source = "outputs/twelfth-selection-v12/phase_B_operational_v1/paired-return-v2/stage-L-paired-v2.json"
    paired = read_json(paired_source)
    paired_rows = []
    for topology_id, artifact in paired["artifacts"].items():
        for workload, measurements in artifact["measurements"].items():
            for measurement in measurements:
                paired_rows.append({
                    "topology_id": topology_id, "model": artifact["model"], "workload": workload,
                    "trial": measurement["trial"], "warmup": measurement["warmup"],
                    "arm": measurement["model"], "pair_order": "/".join(measurement["pair_order"]),
                    "batch": measurement["batch"], "prompt_tokens": measurement["prompt_tokens"],
                    "generated_tokens": measurement["generated_tokens"], "wall_seconds": measurement["wall_seconds"],
                    "peak_allocated_bytes": measurement["peak_allocated_bytes"],
                    "peak_reserved_bytes": measurement["peak_reserved_bytes"],
                })
    write_csv("04_evaluation/results/paired_serving_trials.csv", paired_rows)
    register(ROOT / paired_source, OUT / "04_evaluation/results/paired_serving_trials.csv", "selected-timing-and-memory-fields")

    with (OUT / "SOURCE_MAP.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["project_relative_source", "release_relative_file", "source_sha256_and_transform"])
        writer.writerows(SOURCES)
    files = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name not in {"MANIFEST.sha256", "SOURCE_MAP.csv", "build_release.py"} and ".git" not in p.parts and "__pycache__" not in p.parts and "rebuilt_figures" not in p.parts and p.suffix != ".pyc")
    (OUT / "MANIFEST.sha256").write_text("".join(f"{digest(p)}  {p.relative_to(OUT)}\n" for p in files))
    print(f"source files={len(SOURCES)} release files={len(files)} logical endpoints={len(logical_rows)} question rows={len(quality_rows)} serving trials={len(paired_rows)}")


if __name__ == "__main__":
    main()
