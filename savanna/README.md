# The savanna world: post-training and trait tests

This folder holds the current world and the code that trains and tests on it. One grazing animal moves between a rain pan with reeds (where a lion may lie), a riverbed where a well can be dug, a grassy plain and shade. A herd moves between the plain and the water; it is heard when the animal listens, but never seen. The model reads and writes terse plain-English log lines.

- `world_i1.py`: the simulator, the text, the routine the scripted animal follows, the question chooser, and the exact ideal forecaster (a belief filter over 324 states).
- `train.py`: stage 1. The model learns to predict the printed world lines and to answer forecast questions on days where the routine acts.
- `act.py`, `train_act.py`: stages 2 and 3.
  - In stage 2 the model writes the animal's commands itself.
  - In stage 3 each command is rewarded by how much better the model's own forecasts of the next three hours get than after a second command it could have chosen. Both branches continue with the routine. The reward carries a pull back toward the stage-2 model.
- `battery_i1.py`: the trait tests (access, unified, self-model, metacognition). Each edits one deciding line of a held-out day against a placebo edit, with blind forecasters as floors. `--actor` runs them on days where a trained model chose the commands.
- `seek_test.py`: the seeking test (does the model listen when listening is worth more?) and the own-policy test.
- `fast.py`: the base decoder's forward, with a key/value cache for writing commands.
- `run_seed.sh`: stages 2 and 3 and their tests for one seed on a GPU.

Everything runs in FP32. Nothing the model reads names, describes or asks about a trait.
