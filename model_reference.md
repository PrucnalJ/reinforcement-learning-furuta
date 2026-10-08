# Model reference and MATLAB-to-Python map

## Coordinate convention

The simulator uses

\[
x=[\phi,\dot\phi,\theta,\dot\theta]^T,
\]

where \(\phi\) is the horizontal-arm angle and \(\theta=0\) is the unstable
upright pendulum position. The real Furuta mechanism has one actuator, so the
input is the motor torque \(\tau_\phi\); the pendulum joint is unactuated.

## What changed from the MATLAB demonstration

| MATLAB demonstration | Python starter |
|---|---|
| Script runs one LQR simulation | Environment exposes `reset()` and `step()` |
| Continuous scalar torque from LQR | Integer action mapped to a torque value |
| Forward-Euler integration | Fourth-order Runge--Kutta integration |
| Noise inserted while evaluating dynamics | Optional process noise added after integration |
| State stored as columns of one matrix | State returned at every interaction step |
| No episode definition | Angle limits and maximum duration define an episode |

The nonlinear acceleration equations are otherwise a direct translation of
the supplied function `furuta_pend_sim` with the unactuated torque fixed to
zero.

## Relationship to reinforcement learning

The environment possesses a model because it must simulate what happens after
an action. A tabular SARSA or Q-learning agent can nevertheless be model-free:
it observes a sampled transition

\[
(S_t,A_t,R_{t+1},S_{t+1})
\]

and updates an action-value estimate without calling the dynamics function or
predicting alternative next states. The simulator model and the agent's
knowledge are different concepts.

## Technical reference

Magnus Gäfvert, *Modelling the Furuta Pendulum*, Technical Report TFRT-7574,
Department of Automatic Control, Lund Institute of Technology, 1998.

The paper derives the energy expressions, nonlinear equations of motion, and
the lumped parameters \(\alpha,\beta,\gamma,\delta\). It is useful for students
who want to understand the physics, but it is not required to implement the
tabular learning algorithms.

