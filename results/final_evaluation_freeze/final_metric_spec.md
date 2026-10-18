# Frozen final evaluation metric specification

**Freeze status:** Selected before any final held-out evaluation was run.

## Evaluation unit and conditions

- Dinner scenarios: off_peak, normal, peak, super_peak.
- Frozen environment seeds: 12000–12199 inclusive.
- Every policy uses the same 200 scenario/seed combinations per scenario.
- Keyed exogenous randomness and paired scenario/seed evaluation remain unchanged.
- A fixed baseline has 200 seeds × 4 scenarios = **800 episodes**.
- Each saved learned model has **800 episodes**.
- Each learned architecture retains three training seeds, so its aggregate has **2,400 episodes**.
- No result from seeds 12000–12199 existed when this specification was frozen.

## Primary metrics

1. Episode return (total_reward).
2. Average person waiting time (average_person_wait_ticks; retain tick units and optionally present minutes as ticks × 5 without replacing the tick value).
3. Lost rate: (abandoned_persons + balked_persons) / arrival_persons.
4. Served rate: seated_persons / arrival_persons.

## Secondary metrics

1. Fixed-window seat utilisation.
2. Fixed-window table utilisation.
3. Drain ticks.

Preserve scenario-specific episode return for all four Dinner scenarios.

## Aggregation

- Preserve every per-episode row and every learned model's training seed.
- For each learned architecture, aggregate the three training-seed summaries as **mean ± sample SD** (ddof=1).
- Report per-training-seed summaries before or alongside the architecture aggregate.
- Baselines are fixed policies and are summarized over their 800 paired episodes; do not fabricate a training-seed SD.
- Retain scenario-level summaries as well as the overall summary.

## Paired Best-fit comparison

For each learned model and each identical (scenario, evaluation_seed) pair:

delta_return = learned_model_return - best_fit_return

Report per training seed:

- mean paired delta return;
- percentage better (delta_return > 0);
- percentage equal (delta_return == 0);
- percentage worse (delta_return < 0).

For an architecture, retain all three training-seed results and aggregate transparently; do not collapse the training-seed dimension without showing how it was combined.

## Exclusions and boundaries

- Decision-level Best-fit action agreement is not a mandatory primary metric. It may be reported separately as secondary descriptive evidence only if the existing behaviour-analysis pipeline supports the frozen roster unchanged.
- Episode better/equal/worse must remain separate from decision-level action agreement.
- Party-size metrics are not added: the validated earlier evaluation raw pipeline does not store party-size outcomes, and the existing results collection identified no complete stored party-size outcome table.
- No metric may be added or redefined after final results are observed.
