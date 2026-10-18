# What this package supports

| Goal | Current support | Remaining dependency |
| --- | --- | --- |
| Inspect the exact reported main-study configuration table, 18/96 contrasts, cost events, masks, and paired timings | Included as curated data with source hashes | None for inspection |
| Regenerate the seven manuscript figure layouts | `06_figures/paper/build_paper_figures.py` reads the included frozen A/B statistics | Python with Matplotlib; figures are also included |
| Independently recompute primary accuracy contrasts, bootstrap/sign-flip p-values, and paired serving summaries | `05_statistics/recompute_primary.py` reads only released rows and passed exact comparison with the frozen 18/96 contrasts and 64 artifact-serving summaries | Python with NumPy; original model inference is unnecessary |
| Recreate exact model selection, physical checkpoints, and inference runs from scratch | Source snapshots, frozen protocols, selected masks, model revisions, and output metrics are included | Licensed upstream weights/data, exact sampled calibration/evaluation inputs, portable runtime/requirements, operational configs, and compute hardware; native checkpoint binaries are excluded |
| Recompute historical boundary results cited by the manuscript | `07_boundary/recompute_boundary.py` passed exact checks of two historical Holm-nine quality families, 12 length-diagnostic cells, and 20 Mistral serving cells from the released rows | Original inference and acquisition/deployment materials are not included |

The public dataset should include generated masks, artifact identities and checksums, per-question outcome metrics without benchmark text, paired timing observations, configuration summaries, uncertainty estimates, and cost tables. These are scientific outputs and should be deposited with the code after licensing and disclosure review. Uploading code alone would be insufficient for this paper.

The private audited evidence bundles retain full operational receipts, raw source paths, and native artifacts. They are not interchangeable with the curated public data and are not automatically fit for publication. If exact end-to-end rerunning is claimed, the missing inputs and runnable environment must first be packaged and verified. The public package supports **independent recomputation of the main and historical statistical families and reported figures**; model selection, checkpoint construction, and inference from upstream inputs require further setup.
