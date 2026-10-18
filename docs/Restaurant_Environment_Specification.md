# Restaurant Environment Specification v1.1

Status: Implementation-ready core specification
Purpose: Source of truth for RestaurantEnv implementation
Scope: Core environment only. Baseline/evaluation/reporting choices that do not affect transitions are tracked separately.

## 1. Environment Identity

One Gymnasium environment: RestaurantEnv.

Meal configuration:
- Lunch
- Dinner

Demand scenario:
- Off-Peak
- Normal
- Peak
- Super Peak

The same environment mechanics are used for every meal/scenario combination.

## 2. Physical Restaurant

Tables:
- T1–T4: capacity 2
- T5–T8: capacity 4
- T9–T11: capacity 6
- T12: capacity 8

Total seats = 50.

Tables are fixed in the MVP.

## 3. Time

- One tick = 5 real minutes.
- Seating actions are zero-time decisions.
- Only END_TICK advances real time.
- A decision phase may contain several seating assignments before END_TICK.

## 4. Party Sizes

Allowed party sizes:
1, 2, 3, 4, 5, 6, 7, 8.

A party is never split.

## 5. Compatibility

Allowed party-to-table mappings:
1 → 2,4
2 → 2,4
3 → 4
4 → 4,6
5 → 6
6 → 6
7 → 8
8 → 8

This is an operational modelling assumption.

## 6. Action Space

Discrete(12):
0: 1→2
1: 1→4
2: 2→2
3: 2→4
4: 3→4
5: 4→4
6: 4→6
7: 5→6
8: 6→6
9: 7→8
10: 8→8
11: END_TICK

## 7. Action Selection Semantics

For an action (size s, capacity c):
- Search waiting queue first.
- Among queued size-s parties, choose the longest-waiting party.
- If none exists, search the arrival buffer for size s.
- If several same-size buffer parties exist, choose deterministic lowest party_id.
- Choose the lowest-ID free table of capacity c.
- Generate/lookup that party's dining duration.
- Remove the party from queue/buffer.
- Occupy the table.
- Reward for the assignment step = 0.
- Time does not advance.
- Recompute mask immediately.

END_TICK is always legal.

## 8. Action Mask

A seating action is valid iff:
- at least one matching party exists in queue or buffer, and
- at least one matching-capacity table is free.

END_TICK is always valid.

Required invariants:
- action_mask.sum() >= 1
- masked illegal action passed to the strict environment raises an exception
- epsilon-greedy exploration samples uniformly only from currently valid actions
- TD-target max is computed only over valid next-state actions

All agents and baselines use the same compatibility and mask rules.

## 9. Queue

Hard capacity = 12 waiting parties.

The observation exposes all waiting parties; there is no hidden queue.

Queue ordering:
1. descending waiting time
2. ascending arrival order / party_id
3. padding [0,0] only after all real parties

## 10. Arrival Buffer

Newly generated arrivals enter a temporary current-decision-phase buffer.

Properties:
- waiting time = 0
- no queue-capacity use
- may be seated immediately
- same-size queued party has priority over same-size buffer party
- buffer may not survive END_TICK

On END_TICK:
- let k = free queue slots
- if remaining buffer size <= k, all enter queue
- otherwise each remaining party receives keyed overflow score u_i
- sort ascending by (u_i, party_id)
- first k enter queue
- rest balk
- clear buffer completely

## 11. party_id

party_id is assigned deterministically at generation time:

party_id = (arrival_tick, arrival_index)

It is never reassigned and never depends on policy decisions.

## 12. Arrival Model

N_t ~ Poisson(lambda_t)

lambda_t = meal_base_lambda(t) × scenario_multiplier

Scenario multipliers:
- Off-Peak 0.75
- Normal 1.00
- Peak 1.25
- Super Peak 1.50

Dinner base lambda:
- 17:00–18:00 0.35
- 18:00–19:00 0.50
- 19:00–21:00 0.65
- 21:00–22:00 0.45

Lunch base lambda:
- 11:00–11:30 0.40
- 11:30–12:00 0.75
- 12:00–13:00 1.20
- 13:00–13:30 1.00
- 13:30–14:00 0.60
- 14:00–14:30 0.30

These rates are calibration parameters, not empirical claims.

## 13. Party-Size Distributions

Dinner:
1 15%
2 30%
3 15%
4 20%
5 7%
6 8%
7 2%
8 3%

Lunch:
1 30%
2 40%
3 15%
4 10%
5 2%
6 2%
7 0.5%
8 0.5%

Scenario level does not change party-size composition in the MVP.

## 14. Dining Duration

Dinner mean minutes:
1 45
2 75
3 90
4 90
5 105
6 120
7 135
8 150

Dinner jitter multipliers/probabilities:
0.85/0.05
0.90/0.24
0.95/0.13
1.00/0.25
1.05/0.10
1.10/0.08
1.15/0.10
1.20/0.05

Lunch mean minutes:
1 30
2 45
3 60
4 75
5 85
6 95
7 105
8 115

Lunch jitter:
0.70/0.03
0.85/0.17
0.95/0.25
1.00/0.25
1.10/0.18
1.25/0.12

Actual duration:
- choose keyed multiplier
- multiply by meal/party mean
- round to nearest 5-minute tick using one explicitly chosen rounding policy in code
- duration is only realised when the party is seated
- keyed lookup guarantees the same party gets the same duration under all policies for a fixed test seed

Global maximum supported remaining duration = 36 ticks.

## 15. Patience / Abandonment

Hazard is evaluated after waiting time is incremented during END_TICK.

All bins are left-closed, right-open.

Dinner base hazard:
[0,15) 0.00
[15,30) 0.02
[30,45) 0.05
[45,60) 0.10
[60,75) 0.18
[75,90) 0.28
[90,∞) 0.40

Dinner multiplier:
[17:00,19:00) ×0.8
[19:00,20:00) ×1.0
[20:00,21:00) ×1.6
[21:00,∞) ×2.0
cap = 0.60

Lunch base hazard:
[0,10) 0.00
[10,15) 0.03
[15,20) 0.08
[20,30) 0.15
[30,45) 0.30
[45,∞) 0.50

Lunch multiplier:
[11:00,12:00) ×0.8
[12:00,13:00) ×1.0
[13:00,14:00) ×1.3
[14:00,∞) ×1.6
cap = 0.65

Party size does not modify hazard in the MVP.

Abandonment removes the whole party.

## 16. Reward

R = -0.1 × waited_persons - 1.2 × lost_persons

lost_persons = abandoned_persons + balked_persons

Rules:
- seating assignment = 0 immediate reward
- no seat-waste penalty
- no utilisation reward
- no fixed END_TICK penalty
- a party that abandons after the interval still contributes that interval's waiting cost
- reward is calculated from internal restaurant state/statistics, not reconstructed from the normalized observation

gamma = 1.

## 17. Observation

51 float features.

Block A — tables, 12:
remaining_ticks / 36

Block B — queue, 24:
12 × [party_size/8, min(waiting_ticks/24, 1)]

Block C — arrival buffer, 8:
for size 1–8, normalized count n/(n+1)

Block D — service progress, 1:
Lunch: min(ticks_since_11:00/42, 1)
Dinner: min(ticks_since_17:00/60, 1)

Block E — scenario one-hot, 4

Block F — meal one-hot, 2

Observation space target:
Box(low=0, high=1, shape=(51,), dtype=float32)

## 18. Keyed Randomness

All exogenous random events are keyed.

Suggested conceptual keys:
- arrival count: seed, meal, scenario, tick, event_type
- arrival party size: seed, meal, scenario, tick, arrival_index, event_type
- dining duration: seed, party_id, event_type
- abandonment: seed, party_id, waiting_tick, event_type
- overflow: seed, party_id, tick, event_type

Different policies share exogenous draws but are allowed to reach different states.

## 19. reset()

At meal opening:
- all tables free
- queue empty
- arrival buffer empty
- counters/statistics zero
- choose/receive meal
- choose/receive scenario
- time = meal opening
- return observation + action mask

No customers are generated directly by reset.

The first arrival batch appears after the first END_TICK completes the first 5-minute interval.

## 20. END_TICK

Canonical order:
1. Resolve current arrival buffer into queue or balk.
2. Clear buffer.
3. Advance time one tick.
4. Decrement table remaining times; free tables at zero.
5. Increment every queued party's waiting time one tick.
6. Record waited_persons for the interval.
7. Apply abandonment using updated wait and current meal/time multiplier.
8. Generate arrivals for the interval that has just completed, if it lies within the meal's arrival window.
9. Put new arrivals into a fresh buffer with waiting=0.
10. Compute reward.
11. Build observation and action mask.
12. Check natural termination.

The natural termination condition is also checked after every seating-assignment step.

## 21. Opening / Closing

Lunch arrival window:
[11:00,14:30)

Final lunch arrival interval:
[14:25,14:30)

Dinner arrival window:
[17:00,22:00)

Final dinner interval:
[21:55,22:00)

At closing boundary:
- final interval arrivals are generated normally
- arrivals then become closed
- final buffer must still be processed
- closing does not terminate the world

## 22. Natural Termination

terminated = arrivals_closed AND queue_empty AND buffer_empty

Seated diners may remain.

Their future table-release times no longer influence any future decision or reward.

## 23. Safety Truncation

Safety truncation is an external engineering/diagnostic safeguard, not a restaurant business rule.

Core RestaurantEnv should not create artificial customer-loss penalties merely because a safety limit is reached.

If an external wrapper truncates:
- unresolved waiting customers remain unresolved, not lost
- training code must distinguish truncated from terminated
- the buffer should be empty by construction during legitimate drain-phase truncation

Exact safety-limit value may be chosen during implementation/testing and is not required to build RestaurantEnv v0.1.

## 24. Required Invariants

At minimum:
- action_mask.sum() >= 1
- END_TICK always valid
- no negative waiting time
- no negative remaining dining time
- no party simultaneously in queue, buffer and a table
- one table holds at most one party
- queue length <= 12
- buffer empty after END_TICK buffer-resolution phase
- normalized observation within [0,1]
- normal ended episode: arrivals = seated + abandoned + balked
- truncated episode: arrivals = seated + abandoned + balked + unresolved

## 25. What Is Not Required Before Environment v0.1

The following do not block core implementation:
- final baseline definitions
- utilisation denominator
- final calibration acceptance thresholds
- final choice of extra RL algorithms
- GUI details
- final report wording

These are downstream experimental/evaluation choices.

## 26. Current Implementation Target

RestaurantEnv v0.1 should implement:
- configuration object
- Table entity
- CustomerParty entity
- reset()
- observation creation
- 12-action mask
- seating assignment branch
- END_TICK branch
- queue
- arrival buffer
- keyed arrivals
- keyed dining duration
- abandonment
- reward
- natural termination
- invariants/tests

No RL training in v0.1.
