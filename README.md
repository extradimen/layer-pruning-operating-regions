# Layer-pruning operating regions: code and derived data

This project asks a practical question: **when does deleting entire Transformer layers make a language model faster without making it unusable?** Removing complete blocks produces a smaller, native model, but the choice of *which* blocks to remove may affect reasoning accuracy and difficult cases as much as the number removed. We compare four layer-selection rules across deletion budgets, models, and fixed seeds, then measure task quality, paired inference performance, and cost.

The main experiment uses Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3; interaction-aware combination search, adapted ShortGPT-BI, adapted SLEB iterative selection, and random selection; and `k=1–4` deleted blocks with seeds 11, 23, and 37. Each model also has a shared unpruned (`k=0`) reference. We evaluate PIQA, HellaSwag, GSM8K, C4 loss, and two serving workloads. **The observed speedup rises as more blocks are removed, while GSM8K can lose substantial accuracy after even one removal.** Selection changes the quality and loss-tail outcomes at a fixed budget, so serving speed alone does not identify a useful pruned model. These are findings on a previously observed, project-specific evaluation sample, not a new official test-set result.

This repository accompanies *Where does layer pruning remain useful? A controlled study of selection rules, task capability, serving speed, and tail risk*. It is the paper's **released-data reproducibility package**: readers can inspect selected layers and measured outcomes, recompute the statistical comparisons, and regenerate figures. The released source snapshots do not yet form a turnkey pipeline from upstream model weights to every result; [MODEL_RERUN_GUIDE.md](MODEL_RERUN_GUIDE.md) explains what that requires.

## Choose the task you want to reproduce

| Goal | Start here | What you need |
| --- | --- | --- |
| Inspect masks, quality, timings, costs, and frozen statistics | `02_selection/results/`, `04_evaluation/results/`, `05_statistics/results/` | A CSV/JSON reader; no installation |
| Recompute the main and historical statistics | [QUICKSTART.md](QUICKSTART.md) | Python 3.12 and NumPy; no GPU or model download |
| Regenerate the seven figure layouts | [QUICKSTART.md](QUICKSTART.md) | Python 3.12 and Matplotlib; no GPU |
| Rerun selection, build native pruned models, and evaluate them from upstream weights | [MODEL_RERUN_GUIDE.md](MODEL_RERUN_GUIDE.md) | Additional inputs and runtime configuration **not included** in this repository |

For the released-data path, the shortest setup on macOS or Linux is:

```sh
git clone https://github.com/extradimen/layer-pruning-operating-regions.git
cd layer-pruning-operating-regions
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python verify_release.py
python 05_statistics/recompute_primary.py
python 07_boundary/recompute_boundary.py
python 06_figures/paper/build_paper_figures.py
```

`verify_release.py` checks frozen file hashes and study dimensions. The two recomputation scripts compare their outputs against the included frozen results and report `PASS` when they match. Figures are written to `06_figures/paper/rebuilt_figures/`; the shipped figures stay in `06_figures/paper/figures/`. See [QUICKSTART.md](QUICKSTART.md) for expected counts and output locations. This analysis path does not deploy a service.

## Study design and what is actually counted

The primary matrix uses two seven-billion-parameter instruction models (Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.3), four layer-selection mechanisms, fixed seeds 11/23/37, and whole-block deletion budgets `k=1,2,3,4`. This gives **96 compressed logical endpoints**. A single `k=0` Dense reference is shared within each model; it is not repeated as four mechanisms times three independent seeds. The 96 logical endpoints map to **64 distinct native artifacts**, because some selected masks coincide. No `k=5` result belongs to this matrix.

The common, previously observed E1 evaluation comprises PIQA (128 items), HellaSwag (128), GSM8K (32), and C4 (64 documents). It is not an official test-set confirmation. Quality rows and paired serving observations are released without benchmark text, gold answers, prompts, model generations, or upstream weights. `topology_id` joins question and serving rows to physical artifacts; `02_selection/results/logical_endpoints.csv` maps those artifacts to logical cells. The Dense question rows use `topology_id=DENSE`.

The separate `07_boundary/` analyses cover older `k=6/9`, branch-pruning, and output-length checks. They are not pooled into the main 0–4 whole-layer comparison. The main package includes 23,232 artifact-question rows, 6,400 paired-serving observations, Stage A/B configuration tables (52/196 rows), Holm contrast families (18/96), and cost-event tables (7,874/37,853 rows).

## Repository map

| Directory | Paper workflow | Contents |
| --- | --- | --- |
| `01_design/` | Design freeze | Stage A/B protocols, model revisions, methods, seeds, and source mapping |
| `02_selection/` | Choose layers | Historical and primary source snapshots; selected masks and logical endpoints |
| `03_artifacts/` | Build native pruned models | Construction and reload-check source snapshots; no checkpoint binaries |
| `04_evaluation/` | Measure quality and serving | Scorer/serving source snapshots, per-question outcomes, paired timings |
| `05_statistics/` | Estimate effects and cost | Frozen tables and package-relative primary recomputation script |
| `06_figures/` | Present evidence | Source and PDF/SVG/PNG figures |
| `07_boundary/` | Earlier boundary evidence | Separate frozen results and package-relative recomputation script |

The copied stage modules are historical source snapshots, not a newly wired installer. Several runner files expect a project-specific `spec.json`, upstream material paths, and a GPU environment that are **not shipped**. Do not run those stage scripts as if this repository were a one-command model pipeline. [MODEL_RERUN_GUIDE.md](MODEL_RERUN_GUIDE.md) lists the preserved protocol and the missing pieces needed for that stronger claim. [REPRODUCIBILITY_SCOPE.md](REPRODUCIBILITY_SCOPE.md) states the exact supported scope. `build_release.py` is only for rebuilding the public package inside the original private evidence project, not for public users.

## Provenance and reuse

`SOURCE_MAP.csv` records each copied source file's original project-relative path and SHA-256; `MANIFEST.sha256` freezes the released files. The MIT [LICENSE](LICENSE) covers original project-authored code. It does not relicense upstream models, benchmarks, or underlying text; see [SOURCE_AND_LICENSE_REVIEW.md](SOURCE_AND_LICENSE_REVIEW.md). The native model artifacts and full operational receipts remain private. A persistent archive DOI has not yet been assigned.
