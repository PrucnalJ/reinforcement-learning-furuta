# COSC 527 Project A: Swing-Up and Balancing of a Furuta Pendulum with Tabular SARSA and Q-Learning

@Prucnal-Sachdev and Rocco DeLuca · Oct 11, 2026

**Summary.** Tabular SARSA and Q-learning learned to swing the pendulum up from the bottom, but holding it upright for five seconds required far more experience than reaching it. With 20,000 training episodes, no policy completed a five-second hold in evaluation; SARSA reached upright more often than Q-learning on all three seeds (18 versus 4 of 60 evaluation episodes). Our controlled comparison showed that coarser arm-angle bins learned about twice as fast as the finer bins we had expected to help, and a parameter study showed that our exploration schedule was too slow: with ε fixed at 0.1, SARSA reached upright in 46 of 60. With 50,000 episodes, three policies completed a five-second hold in 1 of 20 evaluation episodes, and the best reached upright in 17 of 20. Zero torque and random actions never reached upright.

## 1. Problem and environment

The agent must swing a simulated Furuta pendulum up from hanging down and hold it within 20° of upright for five consecutive seconds, using three discrete arm torques and learning only from sampled transitions.

The supplied simulator (`furuta_env.py`) integrates Gäfvert's (1998) nonlinear model with a fourth-order Runge–Kutta step at Ts = 0.01 s. Its parameters are not a calibrated QUBE-Servo 3 model, so every claim here is about the simulation. The continuous state is x = \[φ, φ̇, θ, θ̇\]: arm angle, arm angular velocity, pendulum angle, and pendulum angular velocity (rad, rad/s), with θ = 0 upright.

**Verification (`verify_env.py`).** `reset()` returns the four-element state; actions 0, 1 and 2 apply −0.02, 0 and +0.02 N·m, matching `info` (action 3 is rejected); a zero-torque step from the reset state returns reward 0.9993 with both flags False. The supplied file ends an episode at |θ| > 20° or |φ| > π (terminated, reward −10) or after 2000 steps (truncated), and positive torque decreases θ̇. The agent chooses only A\_t; the environment returns S\_t+1, R\_t+1, the terminated and truncated flags and `info`. Because the supplied task is balancing near upright, we wrapped it for swing-up (Section 2) without changing its dynamics or integrator; the instructor approved changing the start and termination conditions (email, October 9, 2026).

## 2. MDP formulation

`swingup_env.py` turns the supplied simulator into the swing-up task in the brief; it sets only the initial condition, reward and episode outcomes. For speed, `fast_furuta.py` evaluates the same equations with scalar arithmetic; over 5000 random steps its trajectory differs from `FurutaEnv` by exactly 0.0.

| Element | Choice |
| --- | --- |
| Initial state | φ = 0, θ = π + U(−5°, 5°), φ̇ and θ̇ each U(−0.05, 0.05) rad/s |
| Angle used by agent and reward | θ̃ = wrapToPi(θ) ∈ \[−π, π) |
| Actions | {−0.02, 0, +0.02} N·m |
| Control period | Agent decides every 5 simulation steps (0.05 s); torque held in between (zero-order hold) |
| Reward per 0.01 s step | cos θ̃ − 0.05·φ² − 1.0·u², summed over the 5 held steps |
| Success | \|θ̃\| ≤ 20° for 500 consecutive simulation steps (5 s); +100 bonus; terminal |
| Unsafe | \|φ\| > π rad, \|φ̇\| > 30 rad/s or \|θ̇\| > 30 rad/s; −1000 penalty; terminal |
| Time limit | 3000 simulation steps (30 s); truncation, not failure |

The upright counter resets whenever the pendulum leaves the 20° region, and leaving it does not end the episode. Success and safety are checked at every 0.01 s simulation step, including the steps inside a held action.

**Why a 0.05 s control period.** At 0.01 s per decision with γ = 0.99 the effective horizon is about 1 s, while a swing-up takes about 5 s of pumping, and the state rarely leaves its bin between decisions. Holding each action for 0.05 s raised the share of tuning episodes reaching upright from about 4% to 19% (Section 8). The simulator still integrates at Ts = 0.01 s.

**Why a −1000 crash penalty.** With −100, the agent learned to crash on purpose (Section 10): while the pendulum hangs down each step costs about −1, so ending the episode early was cheaper than surviving.

## 3. State discretization

The table has 10 × 6 × 14 × 10 = 8400 states and 8400 × 3 = 25,200 state–action entries. Each variable is split by symmetric interior edges; `np.digitize` semantics send values below the first edge to bin 0 and values above the last edge to the final bin, so every real value maps to a valid index.

| Variable | Bins | Positive interior edges (mirrored about 0) | Reason |
| --- | --- | --- | --- |
| φ (rad) | 10 | 0.5, 1.5, 2.2, 2.7 | Two edges near the ±π limit, meant to let the agent see a crash coming (Section 7 tests this) |
| φ̇ (rad/s) | 6 | 1, 4 | Arm still, moving, moving fast |
| θ̃ (deg) | 14 | 5, 12, 20, 45, 90, 135 | Fine near upright for balancing; the 20° edge matches the success region; coarse near the bottom, where only the swing phase matters |
| θ̇ (rad/s) | 10 | 0.5, 1.5, 3, 6 | Separates slow balancing corrections from fast swings; swing-up peaks near 10 rad/s |

The wrapped angle covers \[−π, π), so no pendulum angle falls outside the table. The one seam is at the bottom: ±179° are physically adjacent but fall in bins 13 and 0, which also tells the agent which side it is on.

**Tests (`python discretize.py`).** Values of ±10⁶, values inside a central bin and values exactly on an edge all map to valid indices; 0.1 and 0.1 + 4π rad share a bin; all 8400 rows are reachable; and the fast lookup agrees with the supplied `discretize_state` helper on 20,000 random states. Section 7 compares these arm-angle edges with the first design's ±0.5 and ±1.5 rad.

## 4. Agents, hyperparameters and stopping

Both agents (`agents.py`) share the Q table, the behavior policy and the update step; they differ in one function, the update target. The episode loops follow the Lecture 5 pseudocode (slides 31 and 36).

```latex
\text{SARSA:}\quad Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,[\,R_{t+1} + \gamma\, Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)\,]
```

```latex
\text{Q-learning:}\quad Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha\,[\,R_{t+1} + \gamma \max_a Q(S_{t+1},a) - Q(S_t,A_t)\,]
```

In the code these are `sarsa_target()` and `q_learning_target()`; SARSA selects A\_t+1 with the same ε-greedy policy before updating and then executes it. On the Lecture 5 example (slides 30 and 35) our functions return targets 1.80 and 2.70 and new values 1.16 and 1.34, matching the slides.

- **Terminal transitions** (success or unsafe) use continuation value zero: target = r. A time-limit truncation is not a true terminal state, so the target still bootstraps from S\_t+1.
- **Behavior policy:** ε-greedy. With probability ε a uniformly random action, else the greedy action. **Ties** among maximizing actions are broken uniformly at random, giving each of m tied actions (1 − ε)/m + ε/n (measured 0.451, 0.051, 0.449, 0.050 for slide 28's example).
- **Initialization:** Q(s, a) = 0 for all entries. Since nearly all rewards are negative, zero is optimistic and encourages trying untried actions.

**Hyperparameters.** Learning rate α = 0.1; discount γ = 0.99 per 0.05 s decision (about a 5 s horizon); exploration ε linear from 1.0 to 0.05 over episodes 1–15,000, then 0.05; 20,000 training episodes per run; training seeds 0, 1 and 2 for each method and configuration.

**Stopping criterion.** Training runs for a fixed 20,000 episodes. Fixed budgets keep SARSA, Q-learning and the comparison runs directly comparable; a convergence test on Q was not used because ε keeps exploration non-zero. Each run saves its Q table, visit counts, per-episode logs and full configuration to `results/final/`.

## 5. Training results

In the last 1000 of 20,000 training episodes, SARSA reached upright in 29% of episodes and Q-learning in 23% (means of 3 seeds); no baseline training episode succeeded.

&#91;image: Training curves for the baseline configuration: return, episode duration, success and unsafe rate, and fraction reaching upright, for Q-learning and SARSA, mean of 3 seeds with min-max bands\]

*Figure 1. Baseline training, 500-episode moving averages, mean of seeds 0–2 with min–max band.*

Return first falls (episodes 0–4000) while ε is high and random pushing drives the arm into its stop, then rises from about −1700 to −1370 (SARSA) and −1480 (Q-learning) as the agents learn to swing; episodes grow from 5 s to 7–8 s. Essentially every episode still ends unsafely, and the upright fraction rises sharply only after episode 12,000, once ε falls below about 0.3. With a longer budget of 50,000 episodes (ε decaying over 40,000), Q-learning reached upright in 56% of its last 1000 episodes and completed 8 five-second holds in training, SARSA 126; with coarse arm bins, Q-learning reached 71% and two SARSA seeds completed 110 and 101 holds. These runs change the budget, so they are reported beside, not inside, the controlled comparison. A 20,000-episode run takes about 8 minutes (about 37 hours of simulated time).

## 6. Evaluation results

No 20,000-episode policy held the pendulum for five seconds (0 of 120 evaluation episodes); SARSA reached upright in 6.0 of 20 episodes on average versus 1.3 for Q-learning. Holds appeared only after 50,000 episodes.

Each table was evaluated on the same 20 initial conditions (seeds 100000–100019) with ε = 0, a write-protected Q table and no updates. For reference, applying zero torque (the supplied example) or uniformly random actions never reached upright (0 of 20 each; closest approach 178° and 160° from upright on average).

| Policy (20 episodes per trained seed) | Success | Reached upright | Return, mean ± std | Time to upright (s) | Duration (s) |
| --- | --- | --- | --- | --- | --- |
| Q-learning, 20k episodes, 3 seeds | 0 / 60 | 4 / 60 (seeds: 1, 3, 0) | −1421 ± 136 | 7.4 | 5.9 |
| SARSA, 20k episodes, 3 seeds | 0 / 60 | 18 / 60 (seeds: 4, 6, 8) | −1404 ± 131 | 9.6 | 8.2 |
| Q-learning, 50k episodes, seed 0 | 1 / 20 | 14 / 20 | −1221 ± 376 | 8.8 ± 2.6 | 12.0 ± 5.6 |
| SARSA, 50k episodes, seed 0 | 0 / 20 | 15 / 20 | −1269 ± 52 | 6.0 ± 0.9 | 7.2 ± 2.1 |
| Q-learning, 50k episodes, coarse arm bins, seed 0 | 1 / 20 | 17 / 20 | −1012 ± 514 | 8.5 ± 2.4 | 14.4 ± 7.3 |
| SARSA, 50k episodes, coarse arm bins, seeds 0 and 1 | 1 / 40 | 20 / 40 (seeds: 13, 7) | −1280 ± 445; −1308 ± 62 | 13.5; 7.8 | 19.8; 7.7 |

Return std is the within-seed spread over 20 episodes, averaged over seeds; time to upright averages only the episodes that reached upright. Every unsuccessful 20k-episode policy ended unsafely, in the replayed episodes always by the arm passing ±180°. Two 50k coarse-grid policies survived the full 30 s in 2 of 20 episodes each.

&#91;image: Fraction of evaluation episodes reaching upright and holding for five seconds, for every trained policy\]

*Figure 2. Fraction of the 20 evaluation episodes per policy that reached upright (light) and held for five seconds (green), with counts.*

&#91;image: Wrapped pendulum angle versus time for the first five evaluation episodes of the seed-0 Q-learning and SARSA policies\]

*Figure 3. Wrapped pendulum angle in representative evaluation episodes (seed 0); the green band is the ±20° upright region.* The traces show growing swings over the first 4–8 s. SARSA's swings get closer to 0° before the episode ends at the arm limit.

## 7. Controlled comparison: arm-angle discretization

Changing only the arm-angle bins from 10 (baseline) to 6 roughly doubled how often episodes reached upright, for both methods. This refuted the hypothesis that motivated the finer bins.

**Why this choice.** Replays during tuning showed that every unsafe termination was the arm passing ±180° while the agent pumped the swing. With edges at ±0.5 and ±1.5 rad, one bin spanned 86°–180°, so the agent could not tell a safe arm position from an imminent crash. We hypothesized that adding edges at ±2.2 and ±2.7 rad (10 bins) would let it learn to avoid the stop, and made that the baseline. The comparison runs use the original edges; every other setting, the 20,000-episode budget and the seeds (0–2) are identical.

|  | Q-learning, 10 bins | Q-learning, 6 bins | SARSA, 10 bins | SARSA, 6 bins |
| --- | --- | --- | --- | --- |
| Reached upright, last 1000 training episodes | 23% | 51% | 29% | 55% |
| Evaluation episodes reaching upright (of 60) | 4 | 30 | 18 | 41 |
| Evaluation duration before termination (s) | 5.9 | 10.2 | 8.2 | 9.4 |
| Five-second holds (training / evaluation) | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 0 |

**Interpretation.** The coarse grid has 5040 states instead of 8400, so with the same experience each state is visited about 1.7 times as often and its Q-values are better estimated. The extra detail near the arm limit did not pay for the slower learning within this budget; the finer grid might win with far more episodes, which we did not test. This is the trade-off the brief warns about: a larger table whose entries are rarely visited learns more slowly. With either grid, every unsuccessful evaluation episode still ended unsafely.

## 8. Parameter and exploration study

Besides the controlled comparison, we changed one setting at a time while tuning. Those runs used one seed and short budgets (Q-learning unless noted), so they show the direction of an effect, not its size.

| Setting | Values tried | Reached upright, last 1000 training episodes | Conclusion |
| --- | --- | --- | --- |
| Crash penalty (3000 episodes, 0.01 s decisions) | −100 vs −1000 | Return rose while the agent crashed on purpose; with −1000, SARSA learned to hold still for 30 s instead | −1000 kept |
| Crash penalty (5000 episodes) | −1000 vs −5000 | 7% vs 9% (SARSA 13%); nearly every episode still crashed | No help |
| Decision period (4000 episodes, 6 arm bins) | 0.01 s vs 0.05 s | 4% vs 19% | Largest single gain; 0.05 s kept |
| Discount γ (0.05 s decisions) | 0.99 vs 0.995 | 19% vs 8% at 4000 episodes; 20% vs 18% at 20,000 | 0.99 kept |
| Discount γ (0.01 s decisions) | 0.999 | 4% | No help |
| Learning rate α (20,000 episodes) | 0.1 vs 0.3 | 20% vs 25% | Small gain on one seed; not adopted |
| Arm-position weight λφ (5000 episodes) | 0.05 vs 0.2 | 7% vs 13%; every episode still crashed | Not adopted |
| Training budget | 20,000 vs 50,000 episodes | 20% vs 56%; 0 vs 8 holds in training | Largest effect overall (Sections 5–6) |

**Exploration schedule.** We then tested the exploration schedule the way we ran the controlled comparison: three seeds, 20,000 episodes, every other setting at baseline, and frozen greedy evaluation on the same 20 starts.

| ε schedule | Q-learning: training / evaluation | SARSA: training / evaluation |
| --- | --- | --- |
| 1.0 → 0.05 over 15,000 episodes (baseline) | 23% / 4 of 60 | 29% / 18 of 60 |
| 1.0 → 0.05 over 5000 episodes | 37% / 29 of 60 | 42% / 32 of 60 |
| Fixed at 0.1 | 37% / 15 of 60 | 54% / 46 of 60 |

Training is the share of the last 1000 episodes that reached upright; evaluation counts episodes that reached upright. No schedule produced a five-second hold in evaluation.

Both alternatives beat our baseline schedule for both methods. The baseline acts mostly at random for its first 10,000 episodes (ε > 0.35), and random pushing mostly drives the arm into its stop, so much of that experience is about crashing; with Q starting at 0 and almost every reward negative, untried actions keep getting tried even when ε is small. SARSA gained most from a fixed ε = 0.1 (46 of 60, more than the coarse grid's 41), likely because a constant ε keeps the ε-greedy policy it evaluates from changing under it. Seeds disagree widely (11, 17 and 1 of 20 for Q-learning with the fast schedule), so this ranking is rough. We found it after fixing the baseline; a faster or fixed ε is the first change we would make next.

&#91;image: Four before-and-after bar pairs: decision period 0.01 to 0.05 s took the share of runs reaching upright from 4% to 19%; crash penalty −100 to −1000 took seconds before crashing from 6.7 to 12.8; arm bins 10 to 6 took SARSA from 18 to 41 of 60 evaluation episodes reaching upright; exploration slow decay to fixed 0.1 took SARSA from 18 to 46 of 60\]

*Figure 4. Each change on its own, before (gray) and after (blue): decision period, crash penalty, arm bins and exploration schedule. Evaluation counts are over 60 episodes (3 seeds × 20).*

## 9. Conceptual questions

**(a) Why is the agent model-free?** It only learns by trying moves and seeing what happens. Every update uses just what `env.step()` handed back: the four numbers it saw, the move it made, the points it got, and the four numbers it saw next. It never looks at the physics equations and never predicts what a move it did not try would have done. The simulator has a model; our agent does not use it (Lecture 1, slide 30). The hand-written rules in Section 10 were only checks, not part of either agent.

**(b) Target policies.** SARSA's target policy is the ε-greedy policy it is actually following: its target uses Q(S\_t+1, A\_t+1) for the move it is about to make, random or not. Q-learning's target policy is the greedy policy: its target uses max\_a Q(S\_t+1, a), the best move, whatever it actually does next. The two targets differ whenever the next move is not the greedy one (Lecture 5 example: 1.8 versus 2.7). The short version: SARSA plans around its own mistakes, Q-learning assumes it will not make any. In our runs that made SARSA play safer near the arm limit (6.0 versus 1.3 of 20 reached upright, 8.2 s versus 5.9 s survived) while random moves were still common, and made Q-learning the best agent at 50,000 episodes, once the random moves were nearly gone and its plan was the one actually being played.

**(c) Which states were visited most?** Almost all the practice happened at the bottom of the swing: the two pendulum-angle bins beyond ±135° got about 71% of all updates, while the six bins within ±20°, where balancing is learned, got 0.9% (1.2% for SARSA, 2.1% in the 50,000-episode run). Only 51% of the 25,200 entries were ever updated and 33% at least ten times; the most-updated 1% of entries took 38% of all updates. So most of the table describes spots the agent rarely reaches, and the spots that matter most for the task are the least practiced. That is also why we would not add bins near upright: it would split those few visits further.

&#91;image: Share of all Q updates by bin for each state variable, baseline Q-learning seed 0\]

*Figure 5. Share of all Q-table updates falling in each bin of each state variable (baseline Q-learning, seed 0).*

**(d) Did a larger training return mean a better policy?** No. With the −100 crash penalty, Q-learning's training return rose from −1033 to −669 while every episode still crashed. The score went up because the episodes got shorter: crashing early collected fewer negative per-step rewards than hanging at the bottom. Training return also mixes in random moves and a table that is still changing, so we judged policies only by frozen greedy evaluation (Section 10).

**(e) How does the reward encourage swing-up and balancing?** Every step the agent gets the pendulum's height, cos θ̃, which runs from −1 hanging down to +1 upright. So every bit of extra swing earns points during swing-up, and the most points are exactly where balancing happens. The −0.05φ² term keeps the arm near center and the −u² term is tiny, so height is what matters. The +100 bonus marks the 5 s hold. The crash penalty is −1000 because staying at the bottom costs about −500 over the horizon the agent sees (γ = 0.99 per decision); −100 was cheaper than that, which is why the agent crashed on purpose, and −1000 is clearly worse. −5000 added nothing.

**(f) A limitation for the physical QUBE-Servo 3.** The model has no friction. Gäfvert (1998) reports arm-joint friction of about 0.01 N·m Coulomb and 0.015 N·m stiction, about half of our entire 0.02 N·m push. A policy that learned to pump a frictionless arm would push far less effectively on hardware, so it would not just transfer over. The parameters are also not calibrated to the QUBE, and the real system adds sensor noise and motor delay.

## 10. Limitations and lessons learned

**Lessons from the design process.**

1. **The agent optimizes the reward, not our intent.** With a −100 crash penalty the training return rose while every episode still ended unsafely and episodes got shorter (8.7 s to 6.9 s): crashing early was cheaper than collecting about −1 per step with the pendulum down. A rising return was hiding a worse behavior.
2. **Diagnose before tuning.** A larger penalty (−5000) did not stop the crashes; replaying greedy episodes found the cause, the arm stop. Our fix for it, finer arm bins, was refuted by the controlled comparison (Section 7): the diagnosis was right, the fix was not.
3. **The task is feasible.** A hand-written pumping rule with arm centering reached upright in about 6 s with |φ| ≤ 122°, and a hand-tuned bang-bang rule held the pendulum for 6 s from four perturbed starts, so the failures are learning failures, not physics limits. (These rules were only checks; the submitted policies are learned.)

**Limitations of the tabular approach.**

- **No generalization.** Each of the 25,200 entries is learned separately; experience at θ = 10.0° says nothing about 10.1° unless both fall in the same bin.
- **Resolution versus size.** Doubling the bins of all four variables gives 16 times as many states, most rarely visited.
- **Aliasing.** One bin mixes states that need different actions (an arm at 90° versus 170° in the first design), so the Markov property is only approximate.
- **Sample cost.** 20,000 episodes is about 37 hours of simulated experience per run, far more than is practical on hardware.

These limits motivate function approximation (Lecture 6): features such as sin θ, cos θ and the velocities let nearby states share what was learned.

## 11. Reproducibility

Code: [github.com/PrucnalJ/reinforcement-learning-furuta](https://github.com/PrucnalJ/reinforcement-learning-furuta) (Python 3.12, NumPy, Matplotlib). The code's defaults are the baseline configuration; the README lists the command for every run in this report. Training seeds are 0, 1 and 2; evaluation seeds are 100000–100019; the saved \`.npz\` files in \`results/final/\` hold every Q table, visit count, log and configuration.

```
pip install -r requirements.txt
python verify_env.py; python discretize.py     # Section 1 and 3 checks
python train.py --method qlearning --seed 0 --tag baseline --out results/final
python train.py --method sarsa --seed 0 --phi-edges 0.5,1.5 --tag coarsephi --out results/final
python make_figures.py                         # figures and tables, Sections 5-9
python watch.py results/final/long50kcoarse_qlearning_seed0.npz   # demo
```

## Appendix A. Use of generative AI

We used Claude (Anthropic), through Claude Code, as permitted by the course policy.

**Representative prompts.**

- "Explain in depth how gamma, epsilon and alpha work, and what happens when each one is set too high or too low."
- "How do the bins work, and why does the number of bins change how fast the agent learns?"
- "Why does deciding every 0.05 s instead of every 0.01 s work better with gamma? Why is the crash penalty −1000 and not −800?"
- "Explain the update rule in plain terms. Why does the agent look at the next state instead of just the reward it got?"
- "What is the difference between SARSA and Q-learning, and why did SARSA do better at 20,000 episodes but not at 50,000?"
- "Go through the conceptual questions with us one at a time so we can check our own understanding before answering."
- "Run the exploration-schedule comparison and add a parameter study table to the report."

**How the responses influenced the work.** The assistant wrote most of the code at our direction: the swing-up wrapper (`swingup_env.py`), the fast copy of the simulator, the discretizer, the SARSA and Q-learning agents, the training, evaluation and plotting scripts, and the viewer and playground tools. It ran the training and evaluation runs we asked for, helped us find trends in the results, and drafted this report and the presentation script, which we reviewed and rewrote in our own words. We used it heavily to learn the concepts: what each setting does, how the bins trade detail for practice, and why the update looks at the next state. For Homework 2, we each built a version independently: Jacek the full swing-up version, Rocco a balance-only version that starts upright with a 324-state table, which gave us two different views of the same problem.

**Suggestions verified, corrected or rejected.**

- **Corrected: the crash penalty.** The assistant first set it to −100. Our first run showed the agent crashing on purpose (Section 10), and we raised it to −1000.
- **Rejected: finer arm bins.** The assistant proposed adding arm-angle bins near the limit so the agent could see a crash coming, and made them the baseline. Our controlled comparison refuted it: the coarser bins learned about twice as fast (Section 7). A −5000 penalty was also tried and rejected.
- **Verified:** the fast simulator against the supplied one (difference 0.0), the fast discretizer against the supplied `discretize_state` helper on 20,000 random states, the ε-greedy probabilities against Lecture 5 slide 28, and the update targets against the Lecture 5 worked numbers (1.16 and 1.34).

**Portions with AI assistance.** Most of the source code, the experiment runs, the figures, and the first drafts of this report and the presentation. **What we did ourselves:** chose what to test and what to keep, ran the scripts and checked the outputs against the report, got the instructor's approval for changing the start and termination conditions (Rocco), built a slider playground to see by hand what ε, γ and α do (Jacek), designed the bin reasoning and the independent balance-only Homework 2 version (Rocco), and wrote Section 9 and the presentation in our own words.
