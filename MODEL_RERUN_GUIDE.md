# From-weights model rerun: protocol map and current gaps

**Status:** This is a map for building an independent model rerun, not a tested installation or launch recipe. The current repository can recompute the released statistics and figures. It cannot, by itself, regenerate all selection results, native checkpoints, quality scores, and paired serving measurements from upstream weights. Do not report a from-weights replication on the basis of the released-data checks alone.

## What the frozen protocol fixes

| Component | Frozen information |
| --- | --- |
| Models | Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` (28 blocks, protect 0/27); Mistral-7B-Instruct-v0.3 revision `c170c708c41dac9275d15a8fff4eca08d52bab71` (32 blocks, protect 0/31) |
| Main design | Four selection mechanisms, seeds 11/23/37, `k=1,2,3,4`, one shared Dense baseline per model; 96 logical compressed endpoints and 64 physical artifacts |
| Evaluation | Common E1 item counts: PIQA 128, HellaSwag 128, GSM8K 32, C4 64; E1 was previously observed, not a fresh blind test |
| Full details | `01_design/phase_A_protocol.zh.txt`, `01_design/phase_A_design.json`, `01_design/phase_B_protocol.zh.txt`, `01_design/phase_B_design.json`; selected masks in `02_selection/results/logical_endpoints.csv` |

Consult the frozen protocols before setting up a run; they specify the actual selection, precision, scoring, and serving rules. The model and dataset source links and attribution cautions are in `SOURCE_AND_LICENSE_REVIEW.md`. Obtain any upstream files under their own terms; none are supplied here.

## Scientific stage order and source locations

1. **Prepare inputs and calibration/evaluation IDs.** Verify both pinned model revisions, tokenizer files, and the four benchmark sources. Reconstruct the exact sampled calibration/evaluation inputs and document their hashes and ordering.
2. **Select layers.** The Stage A and B source snapshots are under `02_selection/A/` and `02_selection/B/`. Compare produced masks with `02_selection/results/logical_endpoints.csv`, including shared physical topologies.
3. **Construct native pruned artifacts.** The Stage A and B construction snapshots are under `03_artifacts/A/` and `03_artifacts/B/`. Check reload identity and artifact hashes; the native checkpoint binaries are not released.
4. **Measure quality and paired serving.** The scorer and paired-serving snapshots are under `04_evaluation/A/` and `04_evaluation/B/`. Recreate the same item order, scoring/parsing rules, warmups, pair order, workloads, and per-run isolation. Compare against `04_evaluation/results/`.
5. **Recompute inference and cost summaries.** The frozen output tables are under `05_statistics/results/`. The package-relative `recompute_primary.py` checks statistics from published rows; this step is already runnable through `QUICKSTART.md`.
6. **Regenerate figures.** Use `06_figures/paper/build_paper_figures.py` after statistical outputs are validated.

These are stages, **not executable commands**. Many historical runner files read a neighboring `spec.json` or assume original private project paths; the released source snapshots do not include the required runtime specs. Calling those files directly from this checkout will not reproduce the experiments.

## What is missing for a runnable from-weights release

- A redistributable, checksum-pinned manifest of the exact calibration and evaluation samples or a deterministic acquisition/sampling procedure that reconstructs their content and order. The public metric tables contain IDs and derived outcomes, not benchmark text or answers.
- Portable, audited runtime setup for selection and inference, including the exact compatible GPU/Python/CUDA/PyTorch/Transformers/tokenizer stack and dependency hashes. `requirements-analysis.txt` covers only NumPy and Matplotlib for released-data analysis.
- Sanitized, portable replacements for each required `spec.json`, path mapping, hardware setup, and run launcher. The original execution used project-specific immutable deployments and governed resource control.
- Download/verification instructions for upstream weights and datasets under their source terms, plus an end-to-end smoke test on one model, seed, and deletion budget before running the 96-cell matrix.
- A validated full-run orchestrator with per-stage checks of selected masks, reloadable checkpoints, scorer outputs, paired timing isolation, actual costs, failures, and return-file hashes.

Until those pieces are built and tested, readers can reproduce the **analysis of released measurements**, and can inspect the original method implementation, but cannot claim an exact end-to-end rerun from upstream model weights. Independently implementing the frozen protocol would be a new methodological replication rather than a verified execution of this repository as-is.
