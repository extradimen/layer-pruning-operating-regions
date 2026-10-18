"""Presentation-only figures from the SHA-frozen Stage K statistics output."""

from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qai-matplotlib-stage-K")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/twelfth-selection-v12/statistics-return-v2/statistics.json"
OUTPUT = ROOT / "outputs/twelfth-selection-v12/evidence-v1"
OUTPUT.mkdir(parents=True, exist_ok=True)
payload = json.loads(SOURCE.read_text())
assert payload["status"] == "PASS_K3_COMPLETE_ALL_METHOD_COMPARISON"
assert len(payload["long_table"]) == 52 and len(payload["contrasts"]) == 18

COLORS = {"interaction_tabu": "#0072B2", "shortgpt_bi_adapted": "#E69F00",
          "sleb_iterative_adapted": "#009E73", "random": "#CC79A7"}
MARKERS = {"interaction_tabu": "o", "shortgpt_bi_adapted": "s",
           "sleb_iterative_adapted": "^", "random": "D"}
LABELS = {"interaction_tabu": "Interaction tabu", "shortgpt_bi_adapted": "ShortGPT BI adapted",
          "sleb_iterative_adapted": "SLEB iterative adapted", "random": "Random"}
MODELS = ("qwen", "mistral")
TASKS = ("piqa", "hellaswag", "gsm8k")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.facecolor": "white"})


def save(fig, stem):
    fig.savefig(OUTPUT / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(OUTPUT / (stem + ".svg"), bbox_inches="tight")
    fig.savefig(OUTPUT / (stem + ".png"), dpi=340, bbox_inches="tight")
    plt.close(fig)


rows = [row for row in payload["long_table"] if row["workload"] == "B1S512T128"]
fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
for i, model in enumerate(MODELS):
    dense = next(row for row in rows if row["model"] == model and row["method"] == "Dense")
    for j, task in enumerate(TASKS):
        ax = axes[i, j]
        baseline = dense[task + "_accuracy"]
        ax.axhline(baseline, color="#243447", linestyle="--", linewidth=1.2, label="Dense")
        for k, method in enumerate(COLORS):
            subset = sorted((row for row in rows if row["model"] == model and row["method"] == method), key=lambda row: row["seed"])
            assert len(subset) == 3
            for shift, row in zip((-0.17, 0, 0.17), subset):
                ax.scatter(k + shift, row[task + "_accuracy"], s=36, marker=MARKERS[method],
                           facecolor=COLORS[method], edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_title(model.upper() + " · " + task.upper())
        ax.set_xticks(range(4), ["Tabu", "BI", "SLEB", "Random"])
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#D9E1E8", linewidth=0.6)
        if j == 0:
            ax.set_ylabel("Question accuracy (0–1)")
fig.legend(handles=[Line2D([0], [0], color="#243447", linestyle="--", label="Dense")],
           loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "figure_A_accuracy_all_seeds")

fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
for i, model in enumerate(MODELS):
    dense = next(row for row in rows if row["model"] == model and row["method"] == "Dense")
    for j, workload in enumerate(("B1S512T128", "B4S2048T128")):
        ax = axes[i, j]
        subset = [row for row in payload["long_table"] if row["model"] == model and row["workload"] == workload and row["method"] != "Dense"]
        assert len(subset) == 12
        for row in subset:
            method = row["method"]
            ax.errorbar((row["piqa_accuracy"] - dense["piqa_accuracy"]) * 100,
                        row["speedup"],
                        yerr=[[row["speedup"] - row["speedup_CI"][0]], [row["speedup_CI"][1] - row["speedup"]]],
                        fmt=MARKERS[method], color=COLORS[method], markeredgecolor="white",
                        markersize=6, elinewidth=0.8, capsize=2, alpha=0.85)
        ax.axhline(1, color="#243447", linestyle="--", linewidth=1)
        ax.axvline(0, color="#243447", linestyle=":", linewidth=1)
        ax.set_title(model.upper() + " · " + workload)
        ax.set_xlabel("PIQA accuracy change from Dense (percentage points)")
        if j == 0:
            ax.set_ylabel("Dense/candidate fixed-length wall speedup")
        ax.grid(color="#D9E1E8", linewidth=0.5)
fig.legend(handles=[Line2D([0], [0], marker=MARKERS[m], color=COLORS[m], label=LABELS[m], linestyle="") for m in COLORS],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "figure_B_quality_serving_tradeoff")

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True)
for i, model in enumerate(MODELS):
    ax = axes[i]
    subset = [c for c in payload["contrasts"] if c["model"] == model]
    assert len(subset) == 9
    for y, cell in enumerate(subset):
        method = cell["comparison"].split("-minus-")[1]
        estimate = cell["difference"] * 100
        lower, upper = [value * 100 for value in cell["nominal95CI"]]
        ax.errorbar(estimate, y, xerr=[[estimate - lower], [upper - estimate]],
                    fmt=MARKERS[method], color=COLORS[method], markeredgecolor="#243447" if cell["reject_Holm18"] else "white",
                    markersize=7, elinewidth=1.2, capsize=2.5)
        ax.text(43, y, "Holm p=" + format(cell["p_Holm18"], ".3f"),
                ha="right", va="center", fontsize=7)
    ax.axvline(0, color="#243447", linewidth=1, linestyle="--")
    ax.set_yticks(range(9), [c["task"].upper() + " · " + c["comparison"].split("-minus-")[1].replace("_adapted", "") for c in subset])
    ax.invert_yaxis()
    ax.set_title(model.upper())
    ax.set_xlabel("Tabu minus comparator accuracy (percentage points)")
    ax.set_xlim(-42, 46)
    ax.grid(axis="x", color="#D9E1E8", linewidth=0.6)
fig.tight_layout()
save(fig, "figure_C_holm18_accuracy_contrasts")

leases = payload["cost_lease_lineage"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), width_ratios=[2.1, 1])
for ax, historical in zip(axes, (False, True)):
    items = [item for item in leases if item["shared_or_historical"] == historical]
    labels = [item["role"].replace("_", " ") for item in items]
    values = [item["actual_GPUh"] for item in items]
    colors = ["#D55E00" if item["OSwait"] else ("#56B4E9" if historical else "#0072B2") for item in items]
    ax.barh(labels, values, color=colors, edgecolor="#243447", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("Actual controller lease occupancy (GPU-hours)")
    ax.set_title("Historical shared priors" if historical else "Current Stage K runs, including failure")
    ax.grid(axis="x", color="#D9E1E8", linewidth=0.5)
    for y, value in enumerate(values):
        ax.text(value, y, " " + format(value, ".3f"), va="center", fontsize=8)
    ax.set_xlim(0, max(values) * 1.22)
fig.tight_layout()
save(fig, "figure_D_cost_lease_lineage")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


manifest = {
    "schema_version": 1,
    "project_id": "b85b4b60-c0b8-4cc5-af06-81b4c63eb30a",
    "purpose": "Stage A publication-size evidence figures from frozen statistics; presentation only",
    "presentation_only": True,
    "allowed_derivations": ["percentage-point unit conversion", "candidate-minus-Dense display difference", "fixed method/model ordering"],
    "source": {"path": str(SOURCE.relative_to(ROOT)), "sha256": file_sha(SOURCE), "status": "previously observed E1 follow-up"},
    "script": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": file_sha(Path(__file__))},
    "outputs": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha(path), "bytes": path.stat().st_size}
                for path in sorted(OUTPUT.iterdir()) if path.suffix in (".pdf", ".svg", ".png")],
}
(OUTPUT / "figure_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"figures": 4, "files": len(manifest["outputs"]), "manifest": str((OUTPUT / "figure_data_manifest.json").relative_to(ROOT))}))
