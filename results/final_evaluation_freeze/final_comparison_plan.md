# Frozen final comparison plan

**Freeze status:** Selected before any final held-out evaluation was run.

## Frozen policy roster and research roles

### Baselines

- **Random-valid:** weak valid-action reference.
- **Compatible FIFO:** traditional queue-order reference.
- **Best-fit Greedy:** strong heuristic reference.

### DQN

- **A Default:** default DQN reference.
- **C Shallow-64:** smaller plain DQN selected from development validation.

### PPO

- **A Default:** default PPO reference.
- **C Shallow-64:** similarly sized plain PPO comparator for G.
- **G Cross-Attention:** restaurant-specific party-to-table cross-attention design selected from development validation.

All three training seeds, 31000, 31001, and 31002, are retained for every learned architecture. None is described as a final-test winner.

## Primary preregistered comparisons

### Overall

1. Random-valid vs Compatible FIFO vs Best-fit Greedy.
2. Every learned model vs Best-fit, using the paired episode-return analysis in final_metric_spec.md.

### DQN

1. **C vs A:** Does the smaller plain DQN retain performance?

### PPO

1. **G vs A:** Can the compact attention design match or improve on the large default?
2. **G vs C:** At a similar parameter budget, does cross-attention show an advantage over a plain MLP?

### Secondary descriptive comparison

DQN vs PPO will be reported descriptively. It is not framed as a perfectly controlled architecture comparison because the RL algorithms differ.

## Fixed episode design

- Scenarios: off_peak, normal, peak, super_peak.
- Environment seeds: 12000–12199.
- Identical scenario/seed pairs across every policy.
- 800 episodes per baseline.
- 800 episodes per trained learned model.
- 2,400 episodes per learned architecture across three training seeds.
- 14,400 total policy episodes: 2,400 baseline episodes plus 12,000 learned-model episodes.

## Architecture and training freeze

- No 307,200-step extended training was performed.
- Final learned models use the existing 102,400-step training protocol and their already selected development-validation checkpoints.
- Architecture development ended after G.
- The training-sufficiency audit remains a limitation.
- No architecture, hyperparameter, model checkpoint, model roster entry, primary metric, or primary comparison may change after this freeze.
- No held-out result may be used to add or remove a policy.
