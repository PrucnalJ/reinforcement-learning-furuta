# Furuta pendulum swing-up with tabular SARSA and Q-learning

COSC 527 Midterm Project A. A tabular agent learns, from sampled transitions
only, to swing a simulated Furuta pendulum up from the downward position and
hold it within 20 degrees of upright for 5 s, using three discrete torques.

The simulator (`furuta_env.py`) is the supplied course file, unchanged. Its
parameters come from Gäfvert (1998) and are not a calibrated QUBE-Servo 3
model, so all results concern the simulation.

## Install

Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Files

| File | Purpose |
| --- | --- |
| `furuta_env.py` | Supplied simulator (unchanged) |
| `fast_furuta.py` | Same equations with scalar math for speed; `python fast_furuta.py` shows it matches the supplied simulator exactly |
| `swingup_env.py` | Swing-up task: initial condition, wrapped angle, reward, success / unsafe / time-limit outcomes, 0.05 s control period |
| `discretize.py` | State bins (8400 states x 3 actions); `python discretize.py` runs the range tests |
| `agents.py` | SARSA and Q-learning (Lecture 5 pseudocode), epsilon-greedy with random tie-breaking, greedy evaluation |
| `train.py` | Train one agent; saves Q table, visits, logs and config to an `.npz` |
| `evaluate.py` | Frozen greedy evaluation on fixed seeds |
| `make_figures.py` | Every report figure and table, from `results/final/` |
| `verify_env.py` | Required environment checks |
| `play.py`, `watch.py`, `playground.py` | Play the task yourself, watch a trained agent, or tweak values live |

## Reproduce the reported experiments

The defaults in the code are the baseline configuration.

```bash
python verify_env.py
python discretize.py

# baseline, seeds 0-2, both methods (about 5-10 min per run)
for s in 0 1 2; do
  python train.py --method qlearning --seed $s --tag baseline --out results/final
  python train.py --method sarsa     --seed $s --tag baseline --out results/final
done

# controlled comparison: coarse arm-angle bins, everything else identical
for s in 0 1 2; do
  python train.py --method qlearning --seed $s --phi-edges 0.5,1.5 --tag coarsephi --out results/final
  python train.py --method sarsa     --seed $s --phi-edges 0.5,1.5 --tag coarsephi --out results/final
done

# longer budget (reported separately, not part of the comparison)
python train.py --method qlearning --seed 0 --episodes 50000 --eps-decay-episodes 40000 --tag long50k --out results/final

python make_figures.py          # writes figures/ and figures/summary.json
```

Evaluation uses 20 greedy episodes per table from seeds 100000-100019, identical
for every method and configuration.

The learned Q tables used in the report are committed in `results/final/`.

## Try it

```bash
python play.py                                              # you control the arm (arrow keys)
python watch.py results/final/long50k_qlearning_seed0.npz   # watch a trained agent
python playground.py                                        # sliders for alpha, gamma, epsilon, reward
python train.py --method qlearning --episodes 3000 --live   # watch the agent while it trains
```
