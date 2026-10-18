"""Record one development evening per policy and demand level.

Run: python demo_web/record_traces.py (it switches to the project root itself).
Uses development seed 20000 only. The frozen final test seeds 12000-12199 are never touched.
Output: demo_web/traces.json
"""
import json
import os
import sys
from pathlib import Path
import torch as th

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from restaurant_rl.environment import RestaurantEnv
from restaurant_rl.baselines import choose_best_fit_action, choose_fifo_action
from restaurant_rl.config import END_TICK_ACTION_ID
from restaurant_rl.action_masking import action_mask_from_observation
from restaurant_rl.masked_dqn import MaskedDQN
from sb3_contrib import MaskablePPO

SEED = 20000
assert not (12000 <= SEED <= 12199)
MODELS = {  # training seed whose final mean return is closest to its group's mean
    "DQN A": ("dqn", "models/dqn_default_seed31002_best.zip", 31002),
    "DQN C": ("dqn", "models/dqn_mlp_64_seed31000_best.zip", 31000),
    "PPO A": ("ppo", "models/ppo_default_seed31001_best.zip", 31001),
    "PPO C": ("ppo", "models/ppo_mlp_64_seed31000_best.zip", 31000),
    "PPO G": ("ppo", "models/ppo_cross_attention_seed31001_best.zip", 31001),
}

def pid(party):
    """Format the stable arrival identity used to track a party across replay snapshots."""
    return f"{party.party_id[0]}-{party.party_id[1]}"
def tid(table):
    """Return the occupying party's replay identity, or None for a free table."""
    if table.assigned_party_id is None:
        return None
    return f"{table.assigned_party_id[0]}-{table.assigned_party_id[1]}"
def snapshot(env):
    """Capture the queue, new arrivals and tables needed by the browser replay."""
    return {
        "queue": [
            {"id": pid(party), "size": party.size, "wait": party.waiting_ticks}
            for party in env.waiting_queue
        ],
        "buffer": [
            {"id": pid(party), "size": party.size}
            for party in env.arrival_buffer
        ],
        "tables": [
            {"id": table.table_id, "cap": table.capacity,
             "party": tid(table), "left": table.remaining_ticks}
            for table in env.tables
        ],
    }

def baseline(choose_action):
    """Adapt an observation-and-mask baseline to the recorder's action/probability interface."""
    return lambda observation: (choose_action(observation, action_mask_from_observation(observation)), None)

def learned(kind, path):
    """Use deterministic predictions and record PPO probabilities for display."""
    # Load the selected checkpoint through its existing model class on the CPU.
    model = (MaskedDQN if kind == "dqn" else MaskablePPO).load(path, device="cpu")
    def choose(observation):
        """Predict a legal action and include masked PPO probabilities when available."""
        mask = action_mask_from_observation(observation)
        if kind == "ppo":
            action, _ = model.predict(observation, deterministic=True, action_masks=mask)
            # Read the masked distribution for display; the deterministic prediction above selects the action.
            with th.no_grad():
                observation_tensor, _ = model.policy.obs_to_tensor(observation)
                distribution = model.policy.get_distribution(observation_tensor, action_masks=mask[None])
                probabilities = distribution.distribution.probs[0].cpu().numpy()
            return int(action), {str(i): round(float(probabilities[i]), 4) for i in range(12) if mask[i]}
        action, _ = model.predict(observation, deterministic=True)
        return int(action), None
    return choose

def record(name, choose, scenario):
    """Group seating decisions and their following time update into replay ticks."""
    env = RestaurantEnv(meal="dinner", demand_scenario=scenario)
    observation, info = env.reset(seed=SEED)
    ticks, current_tick = [], {"tick": env.current_tick, "start": snapshot(env), "seats": []}
    total_reward = served = lost = 0.0
    done = False
    while not done:
        action, probabilities = choose(observation)
        assert info["action_mask"][action], "policy chose an illegal action"
        # Save party identities before stepping so the event can be linked to the visible snapshot.
        before = {pid(p): p for p in env.waiting_queue + env.arrival_buffer}
        queued_ids = {pid(p) for p in env.waiting_queue}
        legal = int(info["action_mask"].sum())
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        # Record seating separately because it does not advance simulated time.
        if action != END_TICK_ACTION_ID:
            after = {pid(p) for p in env.waiting_queue + env.arrival_buffer}
            gone = [k for k in before if k not in after][0]
            table = next(t for t in env.tables if tid(t) == gone)
            seat_event = {
                "id": gone, "size": before[gone].size, "table": table.table_id,
                "from": "queue" if gone in queued_ids else "new",
                "dur": table.remaining_ticks, "a": action,
            }
            if probabilities:
                seat_event["probs"] = probabilities
            current_tick["seats"].append(seat_event)
            served += before[gone].size
        else:
            # Use the environment's loss counts and identify which visible parties left this interval.
            remaining_queue_ids = {pid(p) for p in env.waiting_queue}
            lost_ids = [k for k in before if k not in remaining_queue_ids]
            lost += info["abandoned_persons"] + info["balked_persons"]
            current_tick["end"] = {
                "balked": [party_id for party_id in lost_ids if party_id not in queued_ids],
                "abandoned": [party_id for party_id in lost_ids if party_id in queued_ids],
                "lostPersons": info["abandoned_persons"] + info["balked_persons"],
                "waitedPersons": info["waited_persons"], "legal": legal,
            }
            if probabilities:
                current_tick["end"]["probs"] = probabilities
            current_tick["totals"] = {"return": round(total_reward, 2), "served": int(served), "lost": int(lost)}
            # Finish this replay tick with recorded totals, then capture the next starting state.
            ticks.append(current_tick)
            current_tick = {"tick": env.current_tick, "start": snapshot(env), "seats": []}
    # Keep a final seating-only tick when that action naturally ends the episode.
    if current_tick["seats"]:
        current_tick["totals"] = {"return": round(total_reward, 2), "served": int(served), "lost": int(lost)}
        ticks.append(current_tick)
    return {"policy": name, "seed": SEED, "scenario": scenario, "ticks": ticks,
            "final": {"return": round(total_reward, 2), "served": int(served), "lost": int(lost)}}

# Load each policy once, then record the same development seed for all demand levels.
policies = {"Best-fit": baseline(choose_best_fit_action), "FIFO": baseline(choose_fifo_action)}
meta = {"Best-fit": None, "FIFO": None}
for name, (kind, path, training_seed) in MODELS.items():
    policies[name] = learned(kind, path)
    meta[name] = training_seed
out = {"meta": meta, "traces": {}}
for scenario in ["off_peak", "normal", "peak", "super_peak"]:
    for name, choose_action in policies.items():
        trace = record(name, choose_action, scenario)
        out["traces"][f"{name}|{scenario}"] = trace
        print(f"{name:9s} {scenario:10s}", trace["final"])
with (HERE / "traces.json").open("w") as output:
    json.dump(out, output, separators=(",", ":"))
print("wrote", HERE / "traces.json")
