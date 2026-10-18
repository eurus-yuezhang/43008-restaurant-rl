# Final held-out evaluation report

Evaluation started: 2026-10-05T16:27:52+11:00
Evaluation completed: 2026-10-05T16:37:12+11:00

## A. Final held-out protocol

- Dinner scenarios: off_peak, normal, peak, super_peak.
- Frozen environment seeds: 12000–12199 inclusive.
- 800 episodes per fixed baseline or trained model instance.
- All three training seeds retained for every learned architecture.
- Total episodes: 14,400.
- Identical scenario/seed combinations and existing keyed randomness used across policies.
- Random-valid uses the earlier evaluation derivation: NumPy RNG seed = 2,000,000 + environment seed.
- No training, checkpoint selection, tuning, or architecture changes occurred.

## B. Baseline results

| Policy | Return | Wait ticks | Lost rate | Served rate | Seat util. | Table util. | Drain ticks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random-valid | -62.7741 | 3.0114 | 0.1595 | 0.8405 | 0.4597 | 0.5609 | 3.0175 |
| Compatible FIFO | -45.7391 | 2.0764 | 0.1120 | 0.8880 | 0.4948 | 0.5959 | 2.3925 |
| Best-fit Greedy | -42.8098 | 1.9217 | 0.1080 | 0.8920 | 0.5017 | 0.5955 | 2.3287 |

Fixed baselines are summarized over 800 episodes. Their episode-level variability is retained in the CSV files and is not presented as training-seed variability.

## C. DQN A vs C

| Architecture | Return mean ± training-seed SD | Wait | Lost rate | Served rate | Seat util. | Table util. | Drain |
|---|---:|---:|---:|---:|---:|---:|---:|
| DQN A Default | -44.7559 ± 0.6581 | 2.0282 | 0.1126 | 0.8874 | 0.4969 | 0.5939 | 2.3679 |
| DQN C Shallow-64 | -44.2017 ± 0.3608 | 1.9997 | 0.1106 | 0.8894 | 0.4985 | 0.5945 | 2.3621 |

## D. PPO A vs C vs G

| Architecture | Return mean ± training-seed SD | Wait | Lost rate | Served rate | Seat util. | Table util. | Drain |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO A Default | -43.4810 ± 0.4861 | 1.9271 | 0.1115 | 0.8885 | 0.4988 | 0.5942 | 2.2821 |
| PPO C Shallow-64 | -44.7122 ± 1.1534 | 2.0035 | 0.1135 | 0.8865 | 0.4982 | 0.5898 | 2.3746 |
| PPO G Cross-Attention | -43.0536 ± 0.2215 | 1.9312 | 0.1088 | 0.8912 | 0.5010 | 0.5951 | 2.2858 |

## E. Learned agents vs Best-fit

| Architecture | Mean delta return | Better | Same | Worse |
|---|---:|---:|---:|---:|
| DQN A Default | -1.9462 | 788 (32.83%) | 453 (18.88%) | 1159 (48.29%) |
| DQN C Shallow-64 | -1.3920 | 669 (27.88%) | 663 (27.62%) | 1068 (44.50%) |
| PPO A Default | -0.6713 | 272 (11.33%) | 1506 (62.75%) | 622 (25.92%) |
| PPO C Shallow-64 | -1.9025 | 446 (18.58%) | 793 (33.04%) | 1161 (48.38%) |
| PPO G Cross-Attention | -0.2439 | 184 (7.67%) | 1863 (77.62%) | 353 (14.71%) |

Counts pool 2,400 episode pairs per architecture while retaining the three per-model rows in final_paired_vs_bestfit_per_model.csv.

## F. Scenario-specific return

| Policy | Off-peak | Normal | Peak | Super-peak |
|---|---:|---:|---:|---:|
| Random-valid | -22.0985 | -47.7935 | -72.6265 | -108.5780 |
| Compatible FIFO | -11.7880 | -30.3285 | -53.0170 | -87.8230 |
| Best-fit Greedy | -11.6000 | -29.4020 | -49.3960 | -80.8410 |
| DQN A Default | -12.6710 ± 0.4614 | -31.1733 ± 1.2393 | -51.8393 ± 0.3129 | -83.3400 ± 0.7910 |
| DQN C Shallow-64 | -12.4200 ± 0.6938 | -30.7275 ± 0.8760 | -50.7577 ± 0.5550 | -82.9017 ± 0.1516 |
| PPO A Default | -11.6518 ± 0.0988 | -29.6617 ± 0.3218 | -50.3122 ± 0.5712 | -82.2985 ± 1.1291 |
| PPO C Shallow-64 | -12.0765 ± 0.2678 | -30.0170 ± 0.3445 | -50.9052 ± 0.1668 | -85.8503 ± 5.2750 |
| PPO G Cross-Attention | -11.6518 ± 0.0559 | -29.4510 ± 0.1786 | -49.8643 ± 0.4263 | -81.2473 ± 0.4916 |

## G. Main observations

- Among fixed baselines, Best-fit Greedy has the highest mean final return (-42.8098).
- Among learned architectures, PPO G Cross-Attention has the highest mean final return (-43.0536).
- DQN C vs A: challenger mean return is higher by 0.5542; associated wait/loss/service differences are recorded in final_preregistered_comparisons.csv.
- PPO G vs A: challenger mean return is higher by 0.4274; associated wait/loss/service differences are recorded in final_preregistered_comparisons.csv.
- PPO G vs C: challenger mean return is higher by 1.6586; associated wait/loss/service differences are recorded in final_preregistered_comparisons.csv.
- DQN and PPO results are shown together descriptively; they are different algorithms and are not treated as a clean architecture ablation.

## H. Limitations

- The three training seeds are the model-level replication unit; architecture SD is based on three values.
- The training-sufficiency audit remains a limitation of the fixed 102,400-step protocol.
- Party-size outcomes were excluded by the preregistration because the validated evaluator does not provide them unchanged.
- Final held-out results were accepted as observed and were not used for model reselection or tuning.

## Correctness and integrity

- Pre-evaluation tests: 248 passed for both required commands.
- Post-evaluation tests: 248 passed for python -m pytest tests -q (4.39s) and 248 passed for python -m pytest -q (4.50s).
- All 14,400 episodes terminated normally with zero truncations and no illegal actions.
- Arrival/seating/loss accounting and paired exogenous arrival counts passed.
- Source, selected-model, and freeze-file hashes matched their pre-run values.
- Training calls: zero.
