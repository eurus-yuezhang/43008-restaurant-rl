# Project evidence index

| Evidence | Location | Purpose |
|---|---|---|
| Executable notebook | [Final notebook](../notebooks/43008_Restaurant_RL_Final_Executable.ipynb) | Main reader-facing explanation, executable implementation, embedded evidence and plots |
| Environment contract | [Specification](Restaurant_Environment_Specification.md) | Unchanged canonical v1.1 Markdown specification |
| Implementation | [Core package](../restaurant_rl/) | Actual importable code used by saved models, demo and tests |
| Frozen protocol | [Freeze folder](../results/final_evaluation_freeze/) | Plan, metrics, seed freeze/ledger, manifest and authoritative 15-model roster |
| Final evaluation | [Final report](../results/final_heldout_evaluation/final_results_report.md) | Completed held-out results, comparisons, scenario summaries and 14,400 recorded episode rows |
| A–E validation | [A–E](../results/architecture_validation/A-E/) | Architecture comparison, per-seed summary and study manifest |
| F validation | [F](../results/architecture_validation/F/) | Controlled comparison, structured summaries and study manifest |
| F2 validation | [F2](../results/architecture_validation/F2/) | Controlled comparison, dual-structured summaries and study manifest |
| G validation | [G](../results/architecture_validation/G/) | Controlled comparison, summaries, attention diagnostics and study manifest |
| Selected models | [Roster](../results/final_evaluation_freeze/final_model_roster.csv) | Current delivery paths, unchanged model hashes, parameter counts and checkpoint steps |
| Demo | [Web demo](../demo_web/README.md) | Saved seven-policy replay, controls and recording |
| Correctness tests | [Tests](../tests/) | All 27 core pytest modules |
| Runtime and packaging | [Runtime](RUNTIME_AND_DEPENDENCIES.md), [manifest](../SUBMISSION_MANIFEST.md) | Version provenance, included-file hashes and validation results |

Architecture-validation files are unchanged copies reorganized into family folders. The notebook embeds its recorded tables and does not load these CSV files at runtime. Final held-out seeds 12000–12199 were not executed during packaging.

Historical provenance is preserved literally inside frozen manifests and reports. References there to the original archive, runners, raw study runs or omitted checkpoints document how the completed research was produced; those historical paths are not submission runtime dependencies. The cleanup record maps the 15 current model paths to their original names. The current submission manifest lists delivery-file hashes.

Interpretation remains bounded by three training seeds and the recorded training budget. Best-Fit outperforms the strongest learned group on aggregate mean return. G's comparison with F2 supports a possible interaction benefit but does not establish an isolated causal effect of attention. No new evaluation or result regeneration is part of this submission.
