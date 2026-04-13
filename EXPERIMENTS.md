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

## Planned Experiments

### Narrative Arc

Runs 1-5 established the MGR + masking architecture and proved it robust under noise. But all five runs train a single-target agent -- it memorizes one circuit (H→CNOT for |Φ+⟩). The next phase tests **generalization**: can a single policy learn to compile circuits for *multiple* target states, adapting its gate sequence based on what it observes?

This is the transition from "proof-of-concept reward architecture" to "general-purpose RL circuit compiler."

```
Phase 1 (complete): Architecture validation
  Run 1  Failed baseline (Muon)
  Run 2  Lazy-agent trap (absolute reward)
  Run 3  Solved: MGR + masking (noiseless)
  Run 4  Robust under fixed noise
  Run 5  Robust under domain-randomized noise

Phase 2 (next): Multi-target generalization
  Run 6  Multi-Bell: conditional policy across 4 Bell states
  Run 7  Multi-Bell + meta-noise: generalize over targets AND noise

Phase 3 (future): Hardware relevance
  Run 8  Hardware-native gate sets (IBM/Google/IonQ)
  Run 9  3-qubit GHZ (scaling test)
```

---

### Experiment 6: Multi-Bell Conditional Policy

**Status**: Next up
**Goal**: Train a single policy that prepares any of the 4 Bell states, selecting different gate sequences based on the target.

#### Why this matters

Runs 3-5 all converge to the same H→CNOT circuit because the target never changes. The agent is a lookup table with one entry. Experiment 6 forces the agent to become a **conditional circuit compiler**: it must read the target from its observation and output different circuits accordingly.

#### The 4 Bell states and their optimal circuits

| Target | State | Optimal circuit from \|00⟩ | Depth |
|---|---|---|---|
| \|Φ+⟩ | (\|00⟩+\|11⟩)/√2 | H → CNOT | 2 |
| \|Φ-⟩ | (\|00⟩-\|11⟩)/√2 | X → H → CNOT (or H → Z → CNOT) | 3 |
| \|Ψ+⟩ | (\|01⟩+\|10⟩)/√2 | H → CNOT → X | 3 |
| \|Ψ-⟩ | (\|01⟩-\|10⟩)/√2 | X → H → CNOT → X (or H → Z → CNOT → X) | 4 |

The circuits are genuinely different -- the agent must learn 4 distinct strategies and select among them based on the target state information in the observation.

#### Environment changes required

1. **Randomize target state per episode**: sample uniformly from {|Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩} at each `reset()`
2. **Expand observation space**: add the target state's Pauli expectations (15 floats: 6 single-qubit + 9 two-qubit correlators) to the observation, giving a 32-dim obs. Without this, the agent cannot distinguish targets.
3. **MGR and action masking**: unchanged -- they work for any target by construction
4. **Completion threshold**: 0.95 (noiseless); lower for noisy variant

#### Hypotheses

- The agent should discover all 4 optimal circuits
- Convergence will be slower (4x the strategy space, longer optimal circuits for |Ψ-⟩)
- The policy must learn to condition on the target-state correlators in the observation
- 200K-500K timesteps likely needed

#### Open questions

- Is 32-dim obs sufficient, or should the target be encoded more compactly (e.g., a 4-dim one-hot)?
- Does the agent discover the *shortest* circuit for each target, or converge to a longer universal sequence?
- How does entropy behave when the policy must maintain 4 distinct strategies?

---

### Experiment 7: Multi-Bell + Meta-Noise

**Status**: After Experiment 6
**Goal**: Combine multi-target generalization with domain-randomized noise.

This is the capstone of Phase 2: a single policy that prepares any Bell state under any noise regime. If this works, the agent is a genuine general-purpose 2-qubit circuit compiler.

| Parameter | Value |
|---|---|
| Target states | {|Φ+⟩, |Φ-⟩, |Ψ+⟩, |Ψ-⟩}, randomized per episode |
| meta_noise | True |
| Observation | 32-dim (current state + target state Paulis) |
| completion_threshold | 0.80 |
| total_timesteps | 500K-1M |

**Key question**: Under high noise, do different targets require different circuit strategies? Or is the shortest circuit always optimal (less decoherence exposure)?

---

### Experiment 8: Hardware-Native Gate Sets

**Status**: Future (Phase 3)
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
