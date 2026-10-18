# Source and license review for the release candidate

The package contains project-authored Python snapshots and derived metrics, masks, timings, summary tables and figures. It excludes upstream model weights, benchmark prompts, answer labels, generated model text and native checkpoints. This review records upstream attribution; it does not grant redistribution rights to project-authored code or certify rights for derived sample-level data.

| Upstream source | Use in study | Current official source information | Public-package treatment |
| --- | --- | --- | --- |
| [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Base model | Model card lists Apache 2.0 | Weight files excluded; exact revision is in frozen protocol |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Base model | Model card lists Apache 2.0 | Weight files excluded; exact revision is in frozen protocol |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Evaluation | Dataset card lists MIT | Text and answers excluded; derived metrics and sample IDs included |
| [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag) | Evaluation | Dataset card lists MIT | Text and answers excluded; derived metrics and sample IDs included |
| [PIQA](https://huggingface.co/datasets/baber/piqa) | Evaluation | The dataset card used for attribution does not make redistribution terms unambiguous | Text and answers excluded; derived metrics and sample IDs require final review |
| [C4](https://huggingface.co/datasets/allenai/c4) | Calibration and language-model loss | The dataset card does not establish a blanket redistribution license for underlying text | Text excluded; sample IDs and derived NLL require final review |

The author selected MIT for original project-authored code; `LICENSE` states its scope. The sample-level release excludes source text, labels and generated answers. Review its attribution and source terms again when preparing a persistent archive and manuscript submission. The source and license status may change; check the linked primary cards at deposit time.
