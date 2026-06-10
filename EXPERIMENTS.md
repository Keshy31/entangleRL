# EntangleRL Experiment Log

Tracking all training runs, findings, and planned experiments.

---

## Completed Experiments

### Run 1: `mlp_100k_baseline` (Muon Optimizer)

**Date**: 2026-04-13
**Goal**: Baseline 100K-step training with PufferLib 3.0 defaults.

| Parameter | Value |
|---|---|
| Reward | Absolute: `F² - 0.01`, +5.0 bonus at F > 0.95 |
| Optimizer | Muon (PufferLib 3.0 default) |
| Policy | MLP, hidden_size=128 (3.6K params) |
| ent_coef | 0.01, linearly decaying to 0.005 |
| gamma | 0.99 |
| update_epochs | 4 |
| Noise | None (`meta_noise=False`) |
| Envs | 32, 4 workers, Multiprocessing |

**Result**: Complete failure -- no learning.

| Metric | Value |
|---|---|
| Final Fidelity | 0.267 (random baseline) |
| Entropy | 2.197 (maximum, ln(9)) |
| approx_kl | 0.000 |
| importance | 1.000 |
| explained_variance | -0.010 |

**Root cause**: The Muon optimizer (ForeachMuon from heavyball) uses momentum orthogonalization designed for large networks. On a 3.6K-param MLP, it produces zero effective weight updates. The `importance` ratio of exactly 1.000 and `approx_kl` of 0.000 confirm the policy was literally unchanged across all 49 training epochs.

**Model**: `models/mlp_100k_baseline.pth`
**TensorBoard**: `logs/tensorboard/mlp_100k_baseline/`

---

### Run 2: `mlp_100k_adam_baseline` (Adam + Absolute Reward)

**Date**: 2026-04-13
**Goal**: Fix the optimizer, re-test with Adam.

| Parameter | Value |
|---|---|
| Reward | Absolute: `F² - 0.01`, +5.0 bonus at F > 0.95 |
| Optimizer | **Adam** (override PufferLib default) |
| ent_coef | 0.01, linearly decaying to 0.005 |
| gamma | 0.99 |
| update_epochs | 4 |
| anneal_lr | False |
| Noise | None |

**Result**: Learning occurs, but converges to a **lazy-agent local optimum**.

| Metric | Start (2K) | End (100K) |
|---|---|---|
| Fidelity | 0.266 | 0.500 |
| Entropy | 2.197 | 0.025 |
| Identity % | 11.1% | 99.5% |
| explained_variance | -0.017 | -1.190 |

**Root cause**: |00> has intrinsic fidelity 0.5 with the Bell state. The absolute reward `F² - 0.01 = 0.24/step` for doing nothing is higher than the immediate reward for taking the optimal first step (H gate drops F to 0.25, yielding reward 0.0525). The policy entropy collapsed from 2.197 to 0.025 within ~30K steps, permanently locking the agent into Identity-only behavior.

**Key insight**: The fidelity landscape for Bell state preparation is **non-monotonic**. The optimal 2-gate path (H → CNOT) requires passing through F=0.25 to reach F=1.0. Any reward function that penalizes this intermediate dip will create a lazy-agent trap.

**Model**: `models/mlp_100k_adam_baseline.pth`
**TensorBoard**: `logs/tensorboard/mlp_100k_adam_baseline/`

---

### Run 3: `mlp_100k_mgr_masking` (MGR + Action Masking + Tuned Hyperparams)

**Date**: 2026-04-13
**Goal**: Solve the lazy-agent trap with a combined architecture.

| Parameter | Value | Change from Run 2 |
|---|---|---|
| Reward | **Moving-Goalpost (MGR)** | Was absolute F² |
| Action masking | **Self-inverse repeat blocked** | New |
| Optimizer | Adam | Same |
| ent_coef | **0.05 (constant)** | Was 0.01 decaying |
| gamma | **0.95** | Was 0.99 |
| update_epochs | **8** | Was 4 |
| anneal_lr | False | Same |
| Noise | None | Same |

#### MGR Reward Logic
```
At episode start: F_max = F_initial (0.5 for |00> → Bell)

Each step:
  if F_t > F_max:  reward = F_t - F_max;  F_max = F_t
  else:            reward = -0.01
  if F_t > 0.95:   reward += 5.0 (completion bonus, episode terminates)
```

#### Dynamic Action Masking
All 9 gates in the action set are self-inverse (H²=X²=Z²=CNOT²=I²=I). The environment blocks the agent from repeating its immediately previous action. Since PufferLib 3.0 doesn't natively consume `action_mask` from info dicts, this is enforced directly in `step()`: if the agent repeats its last action, the gate application is skipped (equivalent to a wasted no-op step).

**Result**: **Full success** -- agent discovers the optimal Bell state circuit.

| Metric | Start (2K) | Mid (40K) | End (100K) |
|---|---|---|---|
| Fidelity | 0.299 | 0.584 | 0.583 |
| Entropy | 2.157 | 0.802 | 0.944 |
| Episode Length | 17.2 | 1.0 | 1.0 |
| explained_variance | 0.101 | 0.994 | 0.999 |
| value_loss | 0.036 | 0.000 | 0.000 |
| Identity % | 9.0% | 0.0% | 0.1% |

The reported fidelity of ~0.583 is the **average across all timesteps including resets**, not the episode-end fidelity. The agent achieves F=1.0 every episode in 2 steps:
- After reset: info reports F=0.5, step=0
- After H(Q0): info reports F=0.25, step=1
- After CNOT(0→1): info reports F=1.0, step=2 → terminated with +5.0 bonus
- Mean([0.5, 0.25, 1.0]) = 0.583, Mean([0, 1, 2]) = 1.0

Action distribution at convergence: H(Q0)=27.5%, H(Q1)=22.4%, CNOT(0→1)=27.4%, CNOT(1→0)=22.6%. The agent learned **both** valid Bell state circuits (Q0- and Q1-first variants), with a slight preference for the Q0 path.

**Model**: `models/mlp_100k_mgr_masking.pth`
**TensorBoard**: `logs/tensorboard/mlp_100k_mgr_masking/`

---

### Run 4: `fixed_noise` (NISQ-Realistic)

**Date**: 2026-04-13
**Goal**: Test whether the MGR architecture generalizes to noisy quantum simulation.

#### Pre-Experiment Noise Analysis

Ran `python -m src.tools.noise_analysis` with the proposed noise rates. Key findings:

| Metric | Value |
|---|---|
| Noise ceiling (2-gate optimal H→CNOT) | **0.9834** |
| Fidelity loss from noise | 0.0166 |
| Alternative path (H(Q1)→CNOT(1→0)) | 0.9834 (identical by symmetry) |
| Noiseless baseline | 1.0000 |

Depth scan: every additional gate beyond the optimal 2 only decreases fidelity (no compensating sequences exist at these noise levels). Fidelity decays ~0.008 per additional gate step.

| Parameter | Value | Change from Run 3 |
|---|---|---|
| Reward | MGR (same as Run 3) | Same |
| amplitude_damping_rate | 0.05 | Was 0.0 |
| dephasing_rate | 0.02 | Was 0.0 |
| depolarizing_rate | 0.01 | Was 0.0 |
| bit_flip_rate | 0.0 | Same |
| thermal_occupation | 0.0 | Same |
| meta_noise | False | Same |
| gate_time | 0.1 | New |
| completion_threshold | **0.93** | Was 0.95 |
| total_timesteps | **200K** | Was 100K |

**Result**: **Full success** -- agent discovers the same optimal circuit under noise.

| Metric | Start (2K) | Mid (37K) | End (200K) |
|---|---|---|---|
| Fidelity | 0.268 | 0.544 | 0.579 |
| Max Fidelity | 0.602 | 0.668 | 0.662 |
| Entropy | 2.197 | 0.959 | 0.726 |
| Episode Length | 21.0 | 3.5 | 1.0 |
| Completion Rate | 0.0% | 29% | 33% |
| Episode Return | -0.08 | +1.59 | +1.83 |
| explained_variance | 0.018 | -- | 1.000 |

Action distribution at convergence: H(Q0)=49.8%, CNOT(0→1)=50.2%, all others=0.0%. The agent converged to a single Bell circuit variant (Q0-first only), unlike the noiseless run which learned both Q0 and Q1 variants. The entropy of 0.726 (close to ln(2)=0.693) reflects the 2-action alternation.

The reported fidelity of ~0.579 is the time-averaged value across all timesteps including resets: mean(0.5, 0.249, 0.983) ≈ 0.577. The agent achieves F≈0.983 every episode in 2 steps (matching the noise ceiling exactly).

**Key findings**:
- MGR + action masking works identically under fixed noise
- Noise does not introduce any new local optima or deceptive landscapes
- The agent does NOT attempt longer compensating sequences (every extra gate only adds decoherence)
- Convergence speed is comparable to noiseless (~37K steps to reach stable policy vs ~40K noiseless)
- Value function reaches explained_variance=1.000, indicating noise at these levels is deterministic enough for perfect value prediction

**Model**: `models/fixed_noise/final.pt` (+ 7 checkpoints)
**Config**: `models/fixed_noise/config.json`
**TensorBoard**: `logs/tensorboard/fixed_noise/`

---

### Run 5: `meta_noise` (Domain Randomization)

**Date**: 2026-04-13
**Goal**: Train a single robust policy that generalizes across varying noise conditions.

#### Pre-Experiment Analysis

Worst-case noise ceiling (all channels maxed): F=0.918 for optimal 2-gate path. Completion threshold of 0.80 is reachable even in the noisiest episodes.

| Parameter | Value | Change from Run 4 |
|---|---|---|
| Reward | MGR | Same |
| meta_noise | **True** | Was False |
| amplitude_damping_rate | **U(0.0, 0.2)** per episode | Was fixed 0.05 |
| dephasing_rate | **U(0.0, 0.1)** per episode | Was fixed 0.02 |
| depolarizing_rate | **U(0.0, 0.05)** per episode | Was fixed 0.01 |
| bit_flip_rate | **U(0.0, 0.05)** per episode | Was 0.0 |
| thermal_occupation | **U(0.0, 0.1)** per episode | Was 0.0 |
| completion_threshold | **0.80** | Was 0.93 |
| total_timesteps | 200K | Same |
| Noise obs in observation? | **No** (17-dim unchanged) | N/A |

**Result**: **Full success** -- agent discovers optimal circuit despite per-episode noise randomization.

| Metric | Start (2K) | Mid (43K) | End (200K) |
|---|---|---|---|
| Fidelity | 0.261 | 0.562 | 0.568 |
| Max Fidelity | 0.541 | 0.649 | 0.652 |
| Entropy | 2.197 | 0.859 | 0.716 |
| Episode Length | 20.3 | 1.1 | 1.0 |
| Completion Rate | 0.9% | 32% | 33% |
| Episode Return | -0.12 | +1.76 | +1.81 |
| explained_variance | -0.117 | -- | 0.921 |

Action distribution at convergence: H(Q0)=50.2%, CNOT(0→1)=49.8%, all others=0.0%. Same single-variant Bell circuit as Run 4.

**Key findings**:
- The agent does NOT need noise parameters in the observation space. The 17-dim state (Pauli expectations + fidelity + step) is sufficient to learn the optimal policy across the entire noise distribution.
- The policy converges to the same H(Q0)→CNOT(0→1) sequence regardless of noise regime. No episode-specific adaptation occurs -- the optimal circuit is universal.
- explained_variance=0.921 (vs 1.000 in fixed noise). The remaining 8% unexplained variance comes from noise stochasticity across episodes: the value function cannot perfectly predict returns when noise rates vary per episode and are not observed.
- Convergence speed is nearly identical: stable by ~43K steps (vs 37K fixed, 40K noiseless). Domain randomization did not slow learning.
- This validates the "zero-shot transfer" hypothesis: a single policy works across all noise conditions without calibration awareness.

**Open question answered**: Noise parameters in the observation space are NOT needed for this task. The optimal 2-gate circuit is the same regardless of noise severity. However, this may change for harder targets (e.g., 3-qubit GHZ) where different noise regimes might favor different circuit topologies.

**Model**: `models/meta_noise/final.pt` (+ checkpoints)
**Config**: `models/meta_noise/config.json`
**TensorBoard**: `logs/tensorboard/meta_noise/`

---

### Run 3b: `noiseless_mgr` (Re-run with Dense Checkpoints)

**Date**: 2026-06-09
**Goal**: Reproduce Run 3 with `checkpoint_interval=5000` to capture mid-training policy snapshots for the demo video. Also the first run with end-of-episode metrics.

Config identical to Run 3 (MGR + masking, noiseless, 100K steps, seed 42). Training took ~5 minutes; 16 checkpoints saved at ~6K-step intervals under `models/noiseless_mgr/`.

#### End-of-episode metrics (new)

The env now emits `final_fidelity`, `final_max_fidelity`, `episode_completed`, `episode_length`, and `final_return` once per episode (on termination/truncation only). The old per-step keys are time-averaged over every frame including resets, which diluted a perfect 2-step Bell episode to "fidelity 0.583, completed 0.33". The new keys give true per-episode statistics:

| Metric | 2K | 12K | 22.5K | 32.8K | 43K | 100K |
|---|---|---|---|---|---|---|
| F_end | 0.813 | 0.880 | 0.996 | 1.000 | 1.000 | 1.000 |
| Completion | 77% | 84% | 99% | 100% | 100% | 100% |
| Episode length | 26.4 | 25.3 | 6.8 | 2.5 | 2.1 | 2.0 |
| Episode return | +3.97 | +4.38 | +5.40 | +5.49 | +5.49 | +5.49 |

**Key reframing**: even the initial random policy eventually completes 77% of episodes (masking-forced exploration stumbles into the Bell state within ~26 gates). Learning is therefore best described as *circuit compression* -- from ~26 gates to the optimal 2 -- not as "learning to succeed at all". The old time-averaged metrics completely hid this.

#### Checkpoint rollouts (`python -m src.tools.evaluate`)

| Checkpoint | Policy | Behavior |
|---|---|---|
| 6K | sampled | Meandering 7-47 gate episodes, reaches F=1.0 by exploration |
| 24.5K | greedy | Optimal 2-gate circuit already the argmax |
| final | greedy | `H(Q1) -> CNOT(1->0)`, F=1.0000, 100% completion |

Note: this run's greedy policy converged to the **Q1-first** Bell circuit, the mirror of Run 3's Q0-first preference. Both are optimal by symmetry; which variant wins the argmax is seed/run dependent.

**Model**: `models/noiseless_mgr/final.pt` (+ 16 checkpoints)
**Config**: `models/noiseless_mgr/config.json`
**TensorBoard**: `logs/tensorboard/noiseless_mgr/`

---

### Run 6: `multi_bell` (Multi-Bell Conditional Policy)

**Date**: 2026-06-09
**Goal**: Train a single policy that prepares any of the 4 Bell states, conditioning its gate sequence on the target encoded in the observation. First Phase-2 (generalization) run.

#### Environment changes (new, backward-compatible)

- `multi_target=True`: target sampled uniformly from {|Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩} at every `reset()` (`fixed_target_index` pins it for evaluation)
- Observation 17 → 32 dims: `[17:32]` = the target state's 15 Pauli expectations (6 single-qubit + 9 correlators), precomputed once at init
- New metrics: per-step `target_{name}` sampling flags, plus per-target end-of-episode keys `final_fidelity_{name}`, `episode_completed_{name}`, `episode_length_{name}`
- Default single-target mode is unchanged (17-dim obs) -- all earlier checkpoints still load

#### Pre-experiment analysis

Brute-forced minimal circuits over the 8 non-identity gates (sanity-checked in `test_environment.py`):

| Target | F_init from \|00⟩ | Min depth | # optimal circuits |
|---|---|---|---|
| \|Φ+⟩ | 0.5 | 2 | 2 |
| \|Φ-⟩ | 0.5 | 3 | 8 |
| \|Ψ+⟩ | 0.0 | 3 | 8 |
| \|Ψ-⟩ | 0.0 | 4 | 48 |

Two structural notes: (1) the lazy-agent trap only exists for the Φ targets (F_init=0.5); for Ψ targets F_max starts at 0 and any improvement is rewarded. (2) MGR headroom differs per target (0.5 total improvement for Φ, 1.0 for Ψ), so episode returns are target-dependent (+5.49 vs +5.99) and the value function must read the target block of the observation to predict them.

| Parameter | Value | Change from Run 3 |
|---|---|---|
| multi_target | **True** | New |
| Observation | **32-dim** (17 + 15 target Paulis) | Was 17-dim |
| Reward | MGR + action masking | Same |
| completion_threshold | 0.95 | Same |
| total_timesteps | **500K** | Was 100K |
| checkpoint_interval | **10K** | Was 25K |
| Noise | None | Same |
| PPO hyperparams | Adam 3e-4, gamma 0.95, ent_coef 0.05, 8 epochs, hidden 128, seed 42 | Same |

**Result**: **Full success** -- one policy, four distinct minimal-depth circuits. Training took ~5.5 minutes (502K steps).

Per-target episode length / completion over training (end-of-episode metrics):

| Step | Entropy | \|Φ+⟩ | \|Φ-⟩ | \|Ψ+⟩ | \|Ψ-⟩ |
|---|---|---|---|---|---|
| 2K | 2.197 | 32.3 / 68% | 26.0 / 71% | 24.9 / 100% | 24.2 / 92% |
| 51K | 2.084 | 15.1 / 96% | 22.9 / 84% | 18.8 / 89% | 28.3 / 81% |
| 78K | 1.817 | 3.4 / 100% | 13.0 / 98% | 9.8 / 98% | 19.2 / 96% |
| 102K | 0.906 | 2.7 / 100% | 4.0 / 100% | 3.4 / 100% | 5.9 / 100% |
| 131K | 0.590 | 2.2 / 100% | 3.7 / 100% | 3.3 / 100% | 4.4 / 100% |
| 262K | 0.429 | 2.0 / 100% | 3.1 / 100% | 3.0 / 100% | 4.1 / 100% |
| 502K | 0.461 | 2.4 / 99% | 3.0 / 100% | 3.0 / 100% | 4.0 / 100% |

Convergence order followed circuit depth exactly (Φ+ first, Ψ- last). All four targets hit 100% completion by ~102K steps; lengths compress to the optimal 2/3/3/4 by ~260K, giving the optimal mean episode length of 3.0.

#### Greedy rollouts (`evaluate.py --target ...`)

| Target | Discovered circuit | Depth | F_end | Return |
|---|---|---|---|---|
| \|Φ+⟩ | H(Q1) → CNOT(1→0) | 2 (optimal) | 1.0000 | +5.49 |
| \|Φ-⟩ | H(Q1) → CNOT(1→0) → Z(Q1) | 3 (optimal) | 1.0000 | +5.48 |
| \|Ψ+⟩ | X(Q0) → H(Q1) → CNOT(1→0) | 3 (optimal) | 1.0000 | +5.99 |
| \|Ψ-⟩ | X(Q0) → H(Q1) → CNOT(1→0) → Z(Q1) | 4 (optimal) | 1.0000 | +5.98 |

Mixed-target eval (20 episodes, random targets): 100% completion, F_end = 1.0000 ± 0.0000, every episode at minimal depth.

**Key finding -- the policy is compositional, not a 4-entry lookup table.** All four circuits share the Q1-first Bell core H(Q1) → CNOT(1→0); the agent prepends X(Q0) iff the target is a Ψ state (bit flip) and appends Z(Q1) iff it is a minus state (phase flip). The converged action distribution matches this family exactly: H_Q1 33.3%, CNOT_10 33.2%, X_Q0 16.6%, Z_Q1 16.3%, all other gates < 0.3% (per 4 episodes = 12 gates: H1 and CN10 appear 4x each, X0 and Z1 2x each).

**Other findings**:
- Convergence is ~3x slower than single-target (stable by ~130K steps vs ~40K) for 4x the strategy space -- consistent with the hypothesis of slower but tractable learning
- Entropy plateaus at ~0.45 instead of collapsing toward 0: the conditional policy keeps four target-dependent action distributions alive
- explained_variance plateaus at ~0.67 (vs 0.999 single-target), even though the target is observable -- value prediction is harder when returns depend on target identity and position within the circuit

**Open questions answered**:
- 32-dim Pauli encoding of the target is sufficient -- no one-hot needed. The agent conditions on the physical description of the target state.
- The agent discovers the *shortest* circuit for every target (greedy argmax is minimal-depth in all 4 cases), not a longer universal sequence.
- Entropy with 4 strategies stays elevated (~0.45): per-state the policy is near-deterministic, but the marginal over targets keeps it high.

**Model**: `models/multi_bell/final.pt` (+ 49 checkpoints at ~10K intervals)
**Config**: `models/multi_bell/config.json`
**TensorBoard**: `logs/tensorboard/multi_bell/`

---

### Run 7: `multi_bell_meta_noise` (Multi-Bell + Meta-Noise — Phase 2 Capstone)

**Date**: 2026-06-10
**Goal**: One policy that prepares any of the 4 Bell states under domain-randomized noise. `multi_target` and `meta_noise` compose with no env changes; observation stays 32-dim and noise rates remain unobserved.

#### Pre-experiment analysis (`noise_analysis.py --multi_bell`, new mode)

For each target, brute-forced the minimal-depth noiseless-optimal circuits, then ranked every ordering under worst-case noise (all meta-noise rates maxed: ad=0.2, deph=0.1, depol=0.05, bf=0.05, th=0.1) and mean rates (half of max). 200-episode Monte Carlo under sampled meta-noise estimated the completion ceiling at threshold 0.80:

| Target | Depth | # opt | WC best | WC worst | Spread | Mean best | MC comp @0.80 |
|---|---|---|---|---|---|---|---|
| \|Φ+⟩ | 2 | 2 | 0.9181 | 0.9181 | 0.0000 | 0.9595 | 100% |
| \|Φ-⟩ | 3 | 8 | 0.8884 | 0.8785 | 0.0099 | 0.9445 | 100% |
| \|Ψ+⟩ | 3 | 8 | 0.9057 | 0.8779 | 0.0278 | 0.9538 | 100% |
| \|Ψ-⟩ | 4 | 48 | 0.8949 | 0.8400 | 0.0549 | 0.9485 | 100% |

**Gate ordering matters under noise**, and the spread grows with depth (5.5% for |Ψ-⟩). The noise-optimal rule is *stay classical as long as possible*: bit-flips (X) first in the computational basis, H next, the entangling CNOT last, no gates after entanglement (e.g. |Ψ-⟩ best: `X(Q0) → X(Q1) → H(Q0) → CNOT(0→1)`). Notably the |Φ-⟩ noise-best ordering creates the minus phase via `X(Q0) → H(Q0)` (|1⟩ → |−⟩) instead of appending Z after the CNOT — a compositionally different circuit from Run 6's.

Threshold 0.80 clears the worst-case ceiling floor (0.8884) with +0.088 margin → proceed, no contingency. One foreshadowing note: even the *worst* minimal-depth ordering (0.8400) clears 0.80, so the completion bonus cannot discriminate between orderings — only the small MGR fidelity-record differences (~0.01-0.05) push toward noise-optimal circuits.

| Parameter | Value | Change from Run 6 |
|---|---|---|
| multi_target | True | Same |
| meta_noise | **True** (5 channels, U(0, max) per episode) | Was False |
| completion_threshold | **0.80** | Was 0.95 |
| total_timesteps | **1M** | Was 500K |
| checkpoint_interval | 25K | Was 10K |
| Observation | 32-dim, noise rates NOT observed | Same |
| PPO hyperparams | Adam 3e-4, gamma 0.95, ent_coef 0.05, 8 epochs, hidden 128, seed 42 | Same |

Training ran at ~783 SPS (~21 min for 1,001,472 steps), 37 checkpoints.

**Result**: **Success on robustness and conditionality, with a genuine surprise on depth.** All 4 targets reach 100% training completion; but the converged policy is *alphabet-minimal, not depth-minimal*.

Per-target episode length / completion over training (end-of-episode metrics; entropy and explained variance alongside):

| Step | Entropy | expl_var | \|Φ+⟩ | \|Φ-⟩ | \|Ψ+⟩ | \|Ψ-⟩ |
|---|---|---|---|---|---|---|
| 4K | 2.197 | 0.05 | 11.2 / 100% | 43.7 / 14% | 38.7 / 29% | 44.2 / 15% |
| 49K | 2.146 | 0.19 | 26.8 / 56% | 35.1 / 38% | 29.9 / 50% | 36.9 / 33% |
| 100K | 1.696 | 0.57 | 2.0 / 100% | 8.6 / 89% | 11.0 / 84% | 19.9 / 67% |
| 150K | 0.697 | 0.70 | 2.0 / 100% | 3.4 / 99% | **3.1 / 100%** | **4.5 / 99%** |
| 250K | 0.487 | 0.74 | 2.0 / 100% | 3.0 / 100% | **5.0 / 100%** | **5.9 / 100%** |
| 500K | 0.443 | 0.70 | 2.0 / 100% | 3.4 / 99% | 5.0 / 100% | 7.0 / 98% |
| 1M | 0.472 | 0.73 | 2.0 / 100% | 3.0 / 100% | 5.0 / 100% | 6.0 / 100% |

Convergence order again follows circuit depth, and at ~150K **all four targets sat at minimal depth** (2/3/3/4, the Run 6 solution). Then between 150K and 250K the policy *drifted away*: Ψ episode lengths grew from 3.1/4.5 to 5.0/6.0 and stayed there for the remaining 750K steps, with F_end dropping ~0.04 (e.g. Ψ+ 0.941 → 0.898).

#### What the final policy learned: a 3-gate alphabet

Greedy rollouts (identical circuit every episode per target — no per-episode noise adaptation, consistent with Run 5):

| Target | Discovered circuit (final.pt) | Depth | Minimal |
|---|---|---|---|
| \|Φ+⟩ | H(Q1) → CNOT(1→0) | 2 | 2 ✓ |
| \|Φ-⟩ | H(Q1) → CNOT(1→0) → Z(Q1) | 3 | 3 ✓ |
| \|Ψ+⟩ | H(Q1) → CNOT(1→0) → **H(Q1) → CNOT(1→0) → H(Q1)** | 5 | 3 ✗ |
| \|Ψ-⟩ | H(Q1) → CNOT(1→0) → H(Q1) → CNOT(1→0) → H(Q1) → Z(Q0) | 6 | 4 ✗ |

The policy uses only **{H(Q1), CNOT(1→0), Z}** — it eliminated X from its vocabulary entirely. Instead of bit-flipping into the Ψ manifold, it reaches Ψ+ by applying a second `H(Q1) → CNOT(1→0) → H(Q1)` block to Φ+ (an in-place Bell-state rotation), and appends Z for minus phases. Every circuit is unitarily exact: noiseless eval gives **F_end = 1.0000 on all four targets** at depths 2/3/5/6 (zero-shot noise→noiseless transfer holds).

#### Final evals under config noise (greedy, 20 episodes/target)

| Target | Completion | F_end | Len | Mean-noise ceiling (best ordering) |
|---|---|---|---|---|
| \|Φ+⟩ | 100% | 0.9600 ± 0.0120 | 2.0 | 0.9595 (matched exactly) |
| \|Φ-⟩ | 100% | 0.9401 ± 0.0169 | 3.0 | 0.9445 (−0.004, trailing-Z ordering) |
| \|Ψ+⟩ | 100% | 0.9014 ± 0.0263 | 5.0 | 0.9538 (−0.052, two extra gates) |
| \|Ψ-⟩ | **95%** | 0.8526 ± 0.1541 | 8.2 | 0.9485 (−0.07; one bad-draw truncation) |

Mixed-target run (20 episodes, random targets): 100% completion, F_end = 0.9241 ± 0.0442, mean length 4.2. The single |Ψ-⟩ failure is a near-worst-case noise draw where the 6-gate circuit could not cross 0.80; the greedy policy then has no recovery strategy and the masked random-walk decays the state to F=0.19 over 50 steps (completed-episodes-only mean: 0.888).

#### The cost of the long circuits, quantified

Fixed-rate fidelities of the actual learned circuits vs alternatives (via the analysis tool):

| Circuit | Worst-case F | Mean-noise F |
|---|---|---|
| final-1M \|Ψ+⟩ depth 5 | 0.8048 | 0.8991 |
| checkpoint-159K \|Ψ+⟩ depth 3 (`H(Q1)→CNOT(1→0)→X(Q1)`) | 0.8780 | 0.9387 |
| noise-best \|Ψ+⟩ depth 3 | 0.9057 | 0.9538 |
| final-1M \|Ψ-⟩ depth 6 | **0.7712 (below threshold!)** | 0.8800 |
| checkpoint-159K \|Ψ-⟩ depth 4 (`H(Q1)→CNOT(1→0)→X(Q0)→Z(Q0)`) | 0.8400 | 0.9185 |
| noise-best \|Ψ-⟩ depth 4 | 0.8949 | 0.9485 |

The 159K checkpoint — captured during the transient minimal-depth phase — **strictly dominates the final policy on Ψ targets** under noise: Ψ+ 0.946/100% vs 0.901/100%, Ψ- 0.929/100% (depth 4) vs 0.853/95% (depth 6). Training 850K steps past 159K made the policy measurably worse on fidelity, depth, and completion robustness, while consolidating its gate alphabet.

**Key findings**:
- The capstone goal holds: one 5.5K-param policy prepares all 4 Bell states under any noise draw in the meta-noise range, with zero-shot transfer to noiseless. Φ+ tracks its analytic ceiling exactly.
- **Answer to the key question**: physically, the shortest circuit is always optimal and ordering within minimal depth matters (up to 5.5% worst-case). But the agent found *neither* the noise-optimal orderings nor (for Ψ) minimal depth. At threshold 0.80, every minimal ordering *and* the depth-5/6 family complete in ~all training episodes, so the +5.0 bonus provides no gradient between them; the residual ~0.04-0.07 MGR fidelity differences lost to whatever pressure favors reusing one shared action trunk across targets.
- Plausible drivers of the drift (open, not established): the Q1-handed trunk receives gradient from all 4 targets (4x traffic vs target-specific branches); a 3-gate alphabet is easier for the small MLP to represent under per-episode return variance; nothing in the objective penalizes depth once completion saturates (the -0.01 step penalty and γ=0.95 discounting of the bonus were evidently too weak: ~0.5 return difference between depth-3 and depth-5 paths, vs per-episode return noise of similar magnitude under meta-noise).
- explained_variance plateaus at ~0.70-0.74 — essentially Run 6's multi-target level (0.67); stacking unobserved noise on top did not degrade value prediction much further.
- Best artifact for deployment-style use is **not** `final.pt`: `checkpoint_159744.pt` is the better compiler (minimal-depth, X-using, higher noisy fidelity, 100% completion everywhere).

**Open follow-up — Experiment 7b (threshold as depth regularizer)**: raising `completion_threshold` to 0.85 should restore the depth gradient: the learned depth-5/6 circuits fail bad-noise episodes (worst-case 0.80/0.77 < 0.85) while every target's minimal-depth best ordering still clears it (floor 0.8884), and the Monte Carlo minimum F_end of the minimal circuits across sampled draws (0.899+) stays above 0.85. Prediction: same setup at threshold 0.85 converges to depth 2/3/3/4. **Confirmed — see Run 7b below.**

**Model**: `models/multi_bell_meta_noise/final.pt` (+ 37 checkpoints; see `checkpoint_159744.pt` note above)
**Config**: `models/multi_bell_meta_noise/config.json`
**TensorBoard**: `logs/tensorboard/multi_bell_meta_noise/`
**Eval trajectories**: `experiments/run7_eval/` (per-target noisy/noiseless JSON dumps, 159K-checkpoint probes)

---

### Run 7b: `multi_bell_meta_noise_t85` (Threshold as Depth Regularizer)

**Date**: 2026-06-10
**Goal**: Test Run 7's causal hypothesis with a single-variable change: `completion_threshold` 0.80 → **0.85**, everything else identical (same preset, same seed 42, 1M steps).

**Mechanism**: at 0.85, Run 7's learned depth-5/6 Ψ circuits fail bad-noise episodes (worst-case F 0.8048 / 0.7712 < 0.85) while every target's minimal-depth circuits still clear it in all sampled noise draws (worst-case floor 0.8884; MC minimum F_end 0.899). The +5.0 completion bonus therefore discriminates depth again.

**Result**: **Hypothesis confirmed on all three predictions.** Minimal depths restored, X back in the gate alphabet, and — where ordering matters — noise-optimal orderings discovered.

Training (end-of-episode metrics; compare Run 7's drift window):

| Step | Entropy | \|Φ+⟩ | \|Φ-⟩ | \|Ψ+⟩ | \|Ψ-⟩ |
|---|---|---|---|---|---|
| 4K | 2.197 | 28.2 / 50% | 50.0 / 0% | 43.9 / 13% | 50.0 / 0% |
| 100K | 1.836 | 2.1 / 100% | 22.3 / 59% | 15.6 / 74% | 32.5 / 45% |
| 150K | 0.994 | 2.5 / 99% | 5.4 / 95% | 5.2 / 96% | 7.0 / 95% |
| 250K | 0.401 | 2.0 / 100% | 3.0 / 100% | 3.0 / 100% | 4.1 / 100% |
| 500K | 0.432 | 2.0 / 100% | 3.0 / 100% | 3.0 / 100% | 4.0 / 100% |
| 1M | 0.523 | 2.0 / 100% | 3.4 / 99% | 3.0 / 100% | 4.0 / 100% |

Early learning is slower than Run 7 (a random policy crosses 0.85 less often than 0.80 — 0% initial completion on Φ-/Ψ- vs 14-15%), but convergence lands at minimal depth by ~250K and **stays pinned through 1M** — the 150K→250K drift to depth 5/6 never happens.

Greedy rollouts (final.pt, 20 noisy episodes/target, identical circuit every episode):

| Target | Circuit | Depth | F_end (7b) | F_end (Run 7) | Mean-noise ceiling |
|---|---|---|---|---|---|
| \|Φ+⟩ | H(Q0) → CNOT(0→1) | 2 ✓ | 0.9600 ± 0.0120 | 0.9600 | 0.9595 |
| \|Φ-⟩ | H(Q0) → CNOT(0→1) → Z(Q0) | 3 ✓ | 0.9401 ± 0.0169 | 0.9401 | 0.9445 |
| \|Ψ+⟩ | **X(Q1) → H(Q0) → CNOT(0→1)** | 3 ✓ | **0.9546 ± 0.0126** | 0.9014 (d5) | 0.9538 |
| \|Ψ-⟩ | **X(Q1) → CNOT(1→0) → H(Q1) → CNOT(1→0)** | 4 ✓ | **0.9496 ± 0.0131** | 0.8526 (d6, 95%) | 0.9485 |

100% completion on every target (Run 7's Ψ- worst-case failure mode is gone). Mixed 20-episode run: 100%, F_end = 0.9556 ± 0.0211, mean length 3.1. Noiseless eval: F = 1.0000 on all four targets at depths 2/3/3/4.

**Ordering analysis** (the bonus question):
- |Ψ+⟩: the agent's circuit is *exactly* the noise-optimal ordering from the pre-analysis (bit-flip first, entangler last; worst-case 0.9057).
- |Ψ-⟩: a different minimal circuit than the enumeration's top pick, but noise-equivalent — worst-case 0.8947 vs the best ordering's 0.8949 (−0.0002). It obeys the same physics: starts with X, ends on the entangler.
- |Φ-⟩: kept the trailing-Z ordering (worst-case 0.8785 vs best 0.8884). This is the one family where orderings are nearly tied (~1% spread), so the gradient was too weak to matter.
- Dose-response, in other words: the agent honors gate ordering exactly in proportion to how much fidelity it buys (5.5% and 2.8% spreads honored; 1% ignored). Handedness flipped globally vs Run 7 (Q0-core instead of Q1-core) — seed path-dependence between symmetric optima.

**Key findings**:
- One number — the completion threshold — flips the converged solution class from alphabet-minimal (Run 7) to depth-minimal + noise-ordered (Run 7b). The threshold is not just a success criterion; it is the *de facto* depth/fidelity regularizer of the whole reward.
- The MGR fidelity-record increments alone (~0.05) are too weak to shape depth against trunk-reuse pressure; the completion bonus must be placed where the depth difference changes completion probability.
- Practical recipe for future noisy experiments: set the threshold *between* the worst-case ceiling of the minimal circuit and that of the next-longer circuit family, using `noise_analysis --multi_bell` to locate both.
- Phase 2 closes with no caveats: a single 5.5K-param policy compiles all 4 Bell states at minimal depth, with noise-optimal gate ordering where it matters, robust across the full noise distribution, zero-shot to noiseless.

**Model**: `models/multi_bell_meta_noise_t85/final.pt` (+ 37 checkpoints)
**Config**: `models/multi_bell_meta_noise_t85/config.json`
**TensorBoard**: `logs/tensorboard/multi_bell_meta_noise_t85/`
**Eval trajectories**: `experiments/run7b_eval/`

---

## Planned Experiments

### Narrative Arc

Runs 1-5 established the MGR + masking architecture and proved it robust under noise. But all five runs train a single-target agent -- it memorizes one circuit (H→CNOT for |Φ+⟩). Phase 2 tested **generalization**: Run 6 answered yes for the noiseless case (compositional 4-target policy, all minimal-depth); Run 7 added domain-randomized noise and delivered the capstone -- one policy, any Bell state, any noise draw, zero-shot to noiseless -- while exposing that a loose completion threshold lets PPO drift from depth-minimal to *alphabet-minimal* solutions. Run 7b confirmed the cause with a single-variable change: threshold 0.85 restores minimal depth and (where it matters) noise-optimal gate ordering. Phase 2 is complete with no caveats.

This is the transition from "proof-of-concept reward architecture" to "general-purpose RL circuit compiler."

```
Phase 1 (complete): Architecture validation
  Run 1  Failed baseline (Muon)
  Run 2  Lazy-agent trap (absolute reward)
  Run 3  Solved: MGR + masking (noiseless)
  Run 4  Robust under fixed noise
  Run 5  Robust under domain-randomized noise

Phase 2 (complete): Multi-target generalization
  Run 6  Solved: conditional compiler across all 4 Bell states
  Run 7  Capstone: targets x noise; exposed threshold-driven drift to
         alphabet-minimal Ψ circuits
  Run 7b Threshold 0.85 -> depth-minimal restored + noise-optimal orderings

Phase 3 (next): Hardware relevance
  Run 8  Hardware-native gate sets (IBM/Google/IonQ)
  Run 9  3-qubit GHZ (scaling test)
```

---

### Experiment 8: Hardware-Native Gate Sets

**Status**: Next up (Phase 3)
**Goal**: Replace the textbook gate set with hardware-specific native gates.

Real quantum processors don't run {H, X, Z, CNOT}. Each vendor has a native gate set:

| Vendor | Native gates | Notes |
|---|---|---|
| IBM (Eagle/Heron) | √X (SX), Rz(θ), CX | Rz is virtual (zero-cost); SX and CX are physical |
| Google (Sycamore) | √iSWAP, Phased-XZ, CZ | Different entangling gate entirely |
| IonQ (trapped ion) | Rxx(θ) (Molmer-Sorensen), Rz, Ry | All-to-all connectivity |

The environment's `_create_gate_maps()` is already a configurable dictionary. Swapping gate sets is configuration, not architecture. The interesting challenge is **Rz(θ)**: it introduces a continuous parameter, requiring either:
- Discretize angles (e.g., 8 or 16 bins) -- keeps the PPO setup identical
- Hybrid discrete-continuous action space -- more elegant, bigger RL change

The Bell state in IBM native gates requires 4 physical gates: `Rz(π/2) → SX → Rz(π/2) → CX`. The agent would need to discover this non-obvious decomposition from scratch.

**Framing**: "Device-agnostic RL circuit compiler validated on IBM, Google, and IonQ gate sets." This is vendor-neutral and demonstrates the architecture generalizes beyond any single hardware platform.

---

### Experiment 9: 3-Qubit GHZ State (Scaling)

**Status**: Future (Phase 3, requires environment refactoring)
**Goal**: Scale to 3-qubit entanglement.

- Target: GHZ state (|000⟩ + |111⟩)/√2
- Optimal circuit: H(Q0) → CNOT(Q0→Q1) → CNOT(Q0→Q2) = 3 gates
- Initial fidelity |⟨000|GHZ⟩|² = 0.5 (same lazy-agent trap applies)
- Observation space: 3 single-qubit Paulis x 3 qubits = 9, plus 27 two-qubit correlators, plus fidelity + step = 38 dimensions
- Action space: single-qubit gates on 3 qubits + CNOT on 6 qubit pairs + Identity = ~19 actions
- Environment needs `_create_gate_maps()`, `_build_pauli_operators()`, and obs space refactored for variable `num_qubits`

---

## Planned Demo Video (Manim Explainer)

**Status**: Approach decided 2026-06-09, not started.
**Decision**: Manim-style animated explainer, chosen over (a) screen-recording the pygame demo, (b) a headless matplotlib montage MP4, (c) an interactive web demo.

Story beats:
1. The problem: prepare the Bell state from |00⟩; the fidelity landscape is non-monotonic (0.5 → 0.25 → 1.0 along the optimal path)
2. The trap: absolute reward → lazy agent (Run 2 final policy as villain footage: Identity 99.5%)
3. The fix: Moving-Goalpost Reward + action masking, animated on the reward landscape
4. The learning: policy rollouts at successive training checkpoints (source TBD, see below)
5. The payoff: H→CNOT in 2 steps, F=1.000; same circuit survives fixed noise (F=0.983 ceiling) and domain-randomized noise

Prerequisites:
- ~~A rollout-extraction script (`evaluate.py`)~~ **DONE (2026-06-09)**: `python -m src.tools.evaluate --checkpoint <path> --json out.json` dumps per-step actions, gate probabilities, value estimates, fidelities, entanglement, and density matrices. Handles wrapped `.pt` and flat `.pth` checkpoints, greedy or sampled rollouts, `--noiseless` override.
- Mid-training checkpoint source: **DECIDED (2026-06-09) — Option C, both**: re-run `noiseless_mgr` with dense checkpoints (~5K interval) for the main learning arc, plus the existing `fixed_noise`/`meta_noise` checkpoints (7 each, ~26.6K apart) for a noise-robustness act. Four-act structure: the trap (Run 2 cameo) → the learning (noiseless snapshots) → mastery (F=1.0) → robustness under noise (F=0.983 / randomized)
  - ~~Noiseless re-run~~ **DONE (2026-06-09)**: see Run 3b — 16 checkpoints under `models/noiseless_mgr/`, transition window ~12K-43K steps captured
- Run 1 (Muon, ≈random) and Run 2 (lazy agent) final policies as failure-mode cameos — no retraining needed
- TensorBoard curves (entropy, episode length, completion rate) for a synced training-progress scrubber
- New dependency: `manim` (Community Edition) — not yet in requirements.txt

---

## Historical Runs (Pre-Overhaul)

25 historical training runs exist under `logs/tensorboard/` from earlier iterations of the codebase (August 2025). These used RNN policies, different observation spaces, and various hyperparameter sweeps. All exhibited flat fidelity (~0.5) due to insufficient observations and reward structure issues. Listed for reference:

```
quantum_prep/
quantum_prep_rnn_100k_no_noise/
quantum_prep_rnn_100k_no_noise_2/
quantum_prep_rnn_1M/
quantum_prep_rnn_250k_no_noise/
quantum_prep_rnn_250k_no_noise_1e-3_8192_8_42/
quantum_prep_rnn_250k_no_noise_3e-4/
quantum_prep_rnn_250k_no_noise_5e-4/
quantum_prep_rnn_250k_no_noise_7e-4/
quantum_prep_rnn_250k_no_noise_7e-4_16384/
quantum_prep_rnn_250k_no_noise_7e-4_16384_12/
quantum_prep_rnn_250k_no_noise_7e-4_4096/
quantum_prep_rnn_250k_no_noise_9e-4_16384_12/
quantum_prep_rnn_250k_no_noise_9e-4_16384_16/
quantum_prep_rnn_250k_no_noise_9e-4_8192_8/
quantum_prep_rnn_250k_no_noise_9e-4_8192_8_42/
quantum_prep_rnn_300k_no_noise_1e-3_8192_8_42_0.995_a/
quantum_prep_rnn_300k_no_noise_1e-3_8192_8_42_0.995_b/
quantum_prep_rnn_500k/
quantum_prep_rnn_500k_2/
quantum_prep_rnn_500k_no_noise_1e-3_8192_8_42/
quantum_prep_rnn_500k_no_noise_1e-3_8192_8_42_0.995/
quantum_prep_rnn_500k_no_noise_1e-3_8192_8_42_0.995_a/
```
