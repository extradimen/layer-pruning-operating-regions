# Install and recompute the released results

There is no server or web service to deploy for the released-data analysis. On macOS or Linux, install Python 3.12 and Git, then run from a terminal:

```sh
git clone https://github.com/extradimen/layer-pruning-operating-regions.git
cd layer-pruning-operating-regions
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python verify_release.py
```

The two pinned Python packages are for statistical recomputation and figures. `verify_release.py` itself uses only the Python standard library. It checks the frozen package files and ignores local `.git`, `.venv`, caches, and regenerated figures. No GPU or model download is needed for these released-data checks.

Recompute the paper's primary statistics and serving summaries from the released per-question and per-trial tables:

```sh
python 05_statistics/recompute_primary.py
```

Recompute the separately labeled historical boundary analyses:

```sh
python 07_boundary/recompute_boundary.py
```

Each script writes a JSON check result under its `results/` directory and should report `PASS`. The frozen check results are included. In this project's execution environment these resource-intensive analyses were launched through its experiment controller; the commands above describe the script entry points for an independent repository user.

Regenerate the manuscript's seven figure layouts:

```sh
python 06_figures/paper/build_paper_figures.py
```

The 21 generated PDF/SVG/PNG files go to `06_figures/paper/rebuilt_figures/`. This directory is excluded from the frozen manifest, so `python verify_release.py` can be run again afterward. The shipped figures remain in `06_figures/paper/figures/`.

Expected study counts: 96 compressed logical endpoints across two models, four mechanisms, three seeds and `k=1,2,3,4`, plus two shared Dense references. The public main data include 23,232 artifact-question rows and 6,400 paired-serving observations. Historical analyses are separate and must not be pooled into the main 0–4 comparison. See `REPRODUCIBILITY_SCOPE.md` for limits on rerunning upstream model inference.

Full model selection, pruning, and inference require separate installation of the pinned upstream models, datasets, and a compatible GPU runtime. [MODEL_RERUN_GUIDE.md](MODEL_RERUN_GUIDE.md) maps the frozen stages and lists the missing inputs/configuration. This repository preserves the source snapshots and scientific outputs but does not ship a portable inference environment, model weights, or native checkpoints; it makes no one-command full-rerun claim. Do not use `build_release.py` outside the original private evidence project: it reads project-local source paths.
