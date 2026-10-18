# Recompute the released results

Use Python 3.12 and the versions in `requirements-analysis.txt`. Verify the unmodified release first:

```sh
python3 verify_release.py
```

Recompute the paper's primary statistics and serving summaries from the released per-question and per-trial tables:

```sh
python3 05_statistics/recompute_primary.py
```

Recompute the separately labeled historical boundary analyses:

```sh
python3 07_boundary/recompute_boundary.py
```

Each script writes a JSON check result under its `results/` directory and should report `PASS`. The frozen check results are included. In this project's execution environment these resource-intensive analyses were launched through its experiment controller; the commands above describe the script entry points for an independent repository user.

Regenerate the manuscript's seven figure layouts:

```sh
python3 06_figures/paper/build_paper_figures.py
```

The 21 generated PDF/SVG/PNG files go to `06_figures/paper/rebuilt_figures/`. This directory is excluded from the frozen manifest, so `python3 verify_release.py` can be run again afterward. The shipped figures remain in `06_figures/paper/figures/`.

Expected study counts: 96 compressed logical endpoints across two models, four mechanisms, three seeds and `k=1,2,3,4`, plus two shared Dense references. The public main data include 23,232 artifact-question rows and 6,400 paired-serving observations. Historical analyses are separate and must not be pooled into the main 0–4 comparison. See `REPRODUCIBILITY_SCOPE.md` for limits on rerunning upstream model inference.
