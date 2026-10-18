# Restaurant Policy Lab (animated web demo)

An animated replay of recorded dinner evenings. Guests arrive, wait in the queue and walk to the table that the host chooses. Hosts: Best-fit, FIFO, DQN A/C and PPO A/C/G. Demand: Off-Peak, Normal, Peak and Super Peak.

## How it works
1. **Record (Python).** `record_traces.py` runs the real `RestaurantEnv` with each host for one Dinner evening per demand level. It uses development seed 20000 and logs every decision: who was seated where, table timers, arrivals, lost guests, and the per-interval reward. PPO's masked action probabilities are logged too. Learned hosts are loaded with `MaskedDQN.load` / `MaskablePPO.load`, choose actions deterministically under the action mask, and use the training seed closest to each group's final mean (DQN A 31002, DQN C 31000, PPO A 31001, PPO C 31000, PPO G 31001).
2. **Replay (browser).** `src/` holds the page: `page_head.html` (fonts and styles), `page_body.html` (layout) and `page_script.js` (SVG drawing and animation). `build_page.py` puts these together with `traces.json` into one HTML file.

The page replays recorded decisions. It does not run the models live. One evening is a single example; the final results average 800 evenings per model (2,400 per learned architecture).

## Rebuild
From the project root, with the project's Python environment active:

```
python demo_web/build_page.py        # combines the page sources and saved traces
```

Then open `demo_web/restaurant_policy_lab.html` in a browser. A prebuilt copy is included. Page edits only need this build step; `record_traces.py` is the separate recorder for development seed 20000, and is not needed to view or rebuild the saved replay.

## Controls
Space: play/pause · →: next decision · R: restart. Speed 0.5×–4×.
