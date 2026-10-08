# Furuta Pendulum Python Starter

This folder contains a small simulation interface for the reinforcement-learning
project. It is a Python restructuring of the supplied MATLAB file, not a
ready-made RL solution.

## Important model note

The numerical parameters in the original MATLAB file come from Magnus
Gäfvert's 1998 Furuta-pendulum model. They are **not calibrated QUBE-Servo 3
parameters**. Use this code as the course simulation model unless the project
instructions explicitly provide replacement parameters.

The optional paper *Modelling the Furuta Pendulum* explains the derivation of
the nonlinear equations. You do not need to reproduce that derivation to use
the starter environment.

## State, action, and interaction

The continuous state is

```text
x = [arm angle, arm angular velocity,
     pendulum angle, pendulum angular velocity]
```

The upright pendulum corresponds to `pendulum angle = 0`. By default, the
agent chooses among three integer actions:

| Action | Applied torque |
|---:|---:|
| 0 | -0.02 N m |
| 1 | 0 N m |
| 2 | +0.02 N m |

The agent does not need the equations of motion. It interacts with the model
through:

```python
state, info = env.reset()
next_state, reward, terminated, truncated, info = env.step(action)
```

This is model-free learning from the **agent's** perspective even though the
simulator internally uses a known mathematical model to generate transitions.

## Installation and first run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python example_run.py
```

The example applies zero torque. It is only an interface test and is not
expected to balance the pendulum.

## What the starter code supplies

- the nonlinear continuous-time dynamics;
- numerical integration with a fourth-order Runge--Kutta step;
- a discrete torque-action interface;
- episode termination and truncation;
- one baseline reward definition; and
- a helper for converting continuous measurements to discrete table indices.

## What the student must design

The project instructions determine the exact required experiments, but the
starter intentionally does not provide:

- a state discretization;
- a Q table;
- an exploration policy;
- SARSA or Q-learning updates;
- training and evaluation loops; or
- plots and analysis.

Keep training and evaluation separate. During training, exploration may be
enabled and the Q table is updated. During evaluation, freeze the learned
table and normally select greedy actions.

## Suggested checks before training

1. Call `reset()` and verify that a four-element state is returned.
2. Apply each legal action and confirm the torque reported in `info`.
3. Verify that an episode ends when the pendulum leaves the allowed angle.
4. Test the discretizer on values below, inside, and above the selected ranges.
5. Set and report random seeds for reproducible comparisons.

## File map

- `furuta_env.py`: simulator and discretization helper.
- `example_run.py`: minimal interaction example.
- `requirements.txt`: Python dependency.
- `model_reference.md`: notation, MATLAB-to-Python mapping, and paper reference.

