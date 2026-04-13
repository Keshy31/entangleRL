---
tags:
  - ai
  - reinforcementlearning
  - quantum_computing
---
# Comprehensive Guidebook for Reinforcement Learning in Quantum State Preparation

## Introduction

This manual consolidates the concepts, explanations, analogies, and project-specific details for the "Reinforcement Learning for Quantum State Preparation in a Simulated Quantum System" project. It serves as a self-contained primer, playbook, and reference guide.

The project: Using RL (PPO via PufferLib 3.0) to prepare quantum states (e.g., Bell state) in a simulated environment (QuTiP wrapped in Gymnasium), with Pygame for visualization. No physical quantum hardware needed; everything runs in Python on GPU. Training requires Linux or WSL because PufferLib's multiprocessing backend uses `fork()`, which is unavailable on native Windows. See the [README prerequisites](../README.md#prerequisites) for WSL and virtual environment setup.

## 1. Quantum Computing Fundamentals

Quantum computing leverages quantum mechanics to perform computations that are infeasible classically. At its core, it uses qubits instead of bits, enabling parallelism through superposition and entanglement.

### 1.1 Qubits and Basic Principles
- **Qubits vs. Classical Bits**: Classical bits are binary (0 or 1). Qubits can be in a superposition -- a probabilistic mix of 0 and 1 (like a spinning coin). Measurement collapses the superposition to a definite outcome.
  - Analogy: A classical computer checks doors one by one; quantum explores all simultaneously but uses interference to highlight the answer.
- **Superposition**: A qubit state is |psi> = a|0> + b|1>, where |a|^2 + |b|^2 = 1. E.g., a = b = 1/sqrt(2) is the |+> state.
- **Measurement Collapse**: Measuring gives 0 with prob |a|^2, 1 with |b|^2, destroying superposition.
- **Phases and Interference**: Amplitudes have signs/phases. Constructive interference amplifies desired outcomes; destructive cancels others.

### 1.2 State Representations
- **Single Qubit**: 2D complex vector. Bloch sphere: 3D globe where north pole = |0>, south = |1>, equator = superpositions.
- **Multi-Qubit**: Tensor product (e.g., 4D for 2 qubits: |00>, |01>, |10>, |11>). Entangled states cannot be separated.
- **Density Matrices**: Square matrix representation, essential for mixed/noisy systems. Diagonals = probabilities; off-diagonals = coherences (entanglement signature).

### 1.3 Quantum Gates and Operations
Gates transform states reversibly (unitary matrices):
- **Hadamard (H)**: Creates superposition (|0> to |+>). Rotates from pole to equator on Bloch sphere.
- **X (Bit Flip)**: Swaps |0> and |1>. 180-degree rotation around x-axis.
- **Z (Phase Flip)**: Flips sign on |1>. 180-degree rotation around z-axis.
- **CNOT**: Flips target qubit if control qubit is |1>. With H, creates entanglement (Bell state).
- **Sequences**: H on Q0 + CNOT(Q0->Q1) = Bell state (|00>+|11>)/sqrt(2).

### 1.4 Advanced Concepts
- **Entanglement**: Qubits linked non-locally. Bell state: measuring one determines the other. Detectable via two-qubit correlators (<XX>, <YY>, <ZZ>) but invisible to single-qubit measurements.
- **Fidelity**: Similarity score (0-1) between two quantum states. Our success metric.
- **Noise/Decoherence**: Environmental interactions degrade states. Simulated via collapse operators in QuTiP (amplitude damping, dephasing, depolarizing, bit-flip).
- **Partial Observability**: Agent cannot access the full quantum state without collapsing it. Instead observes expectation values of Pauli operators.

## 2. Reinforcement Learning Foundations

RL is decision-making via trial-and-error, with an agent learning to maximize cumulative rewards.

### 2.1 Core Components
- **Agent**: Neural network policy mapping observations to action probabilities.
- **Environment**: QuantumPrepEnv (Gymnasium wrapper around QuTiP simulation).
- **Observations**: 17-dimensional vector: 6 single-qubit Pauli expectations, 9 two-qubit correlators, current fidelity, normalized step count.
- **Actions**: 9 discrete gates (H, X, Z per qubit, CNOT both directions, Identity).
- **Rewards**: Absolute fidelity minus step penalty, with completion bonus.
- **Episode**: One preparation attempt (max 50 steps or fidelity > 0.95).

### 2.2 Policy (Actor)
- Maps observations to action probabilities (stochastic for exploration).
- Entropy coefficient controls exploration vs exploitation (decayed over training).

### 2.3 Value Function (Critic)
- Estimates expected future rewards from a given observation.
- Used to compute advantages (A = actual return - predicted value) for policy updates.
- Gamma (discount factor) = 0.99 balances short-term and long-term rewards.

### 2.4 PPO Algorithm
- Actor-critic with clipped surrogate objective for stable updates.
- Handles noisy environments via advantage normalization.
- Entropy bonus encourages exploration early in training.
- Key hyperparameters: learning_rate=3e-4, clip_coef=0.2, update_epochs=4, batch_size=2048.

### 2.5 Training Loop
- Structure: evaluate() collects rollouts from 32 parallel environments -> train() computes PPO updates and returns logs -> repeat.
- PufferLib handles vectorization, async resets, and done-masking.
- TensorBoard monitors fidelity, losses, action distribution, and SPS.

## 3. Project Integration: Quantum + RL

### 3.1 Environment Design
QuantumPrepEnv is the core Gymnasium environment:
- **Observation**: 17 floats -- single-qubit Paulis, two-qubit correlators (essential for seeing entanglement), fidelity, and step progress.
- **Reward (Moving-Goalpost)**: Only rewards new fidelity records per episode (`F_t - F_max` if improving, -0.01 otherwise), plus +5.0 completion bonus when fidelity exceeds `completion_threshold` (configurable, default 0.95). This prevents the lazy-agent trap caused by non-monotonic fidelity landscapes.
- **Action Masking**: All gates are self-inverse, so repeating the previous action is blocked (equivalent to undoing it). Enforced directly in `step()`.
- **Noise**: Configurable amplitude damping, dephasing, depolarizing, bit-flip, and thermal channels via QuTiP's mesolve. Meta-noise mode randomizes all rates per episode for domain randomization.
- **Completion threshold**: Configurable per experiment. Set lower under noise to ensure the bonus is reachable (e.g., 0.93 for fixed noise, 0.80 for meta-noise).
- **Why two-qubit correlators matter**: The Bell state has <X0>=<Y0>=<Z0>=<X1>=<Y1>=<Z1>=0, identical to a maximally mixed (random) state. Only two-qubit correlators (<XX>=+1, <YY>=-1, <ZZ>=+1) distinguish it.

### 3.2 Training System
- **Configurable experiment runner** (`src/training/train.py`): driven by named presets or CLI args. Supports `--experiment noiseless_mgr|fixed_noise|meta_noise` with full override capability for all env and training parameters.
- PufferLib 3.0 with PuffeRL class. Wraps QuantumPrepEnv via GymnasiumPufferEnv.
- MLP policy (pufferlib.models.Default, hidden_size=128).
- 32 parallel environments, 4 multiprocessing workers.
- Adam optimizer (PufferLib's default Muon fails on small networks).
- Entropy coefficient 0.05 (constant, high floor prevents premature collapse).
- **Logging**: TensorBoard with dynamic environment metric discovery, hparams plugin for cross-experiment comparison, console progress output every ~10 seconds.
- **Checkpointing**: periodic model saves (every 25K steps) with full config JSON alongside the weights.

### 3.3 Tools
- **Noise analysis** (`python -m src.tools.noise_analysis`): computes the fidelity ceiling under specified noise rates by running the optimal gate sequence. Reports achievable fidelity, depth scan for compensating sequences, and recommended completion threshold. Run this before any noisy training experiment.

### 3.4 Visualization and Inference
- Pygame engine loads trained MLP policy (matching hidden_size=128).
- Renders Bloch spheres (QuTiP/Matplotlib), gate probability bars, fidelity meter.
- AI-only mode: argmax actions for deterministic inference.

### 3.5 Workflow

**Platform requirement**: PufferLib's multiprocessing backend requires the `fork` start method (Linux/macOS only). Windows users must run everything inside WSL (Windows Subsystem for Linux) -- see the [README](../README.md#prerequisites) for setup steps.

1. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .entangleRL-env
   source .entangleRL-env/bin/activate
   pip install -r requirements.txt && pip install -e .
   ```
   Re-activate with `source .entangleRL-env/bin/activate` each time you open a new terminal.
2. Sanity check: `python -m src.tests.test_environment`
3. Analyze noise ceiling (if running noisy experiment): `python -m src.tools.noise_analysis`
4. Train: `python -m src.training.train --experiment fixed_noise` (or any preset / custom args)
5. Monitor: `tensorboard --logdir=logs/tensorboard`
6. Visualize: `python -m src.visualization.engine` (WSL users need WSLg or an X server for the Pygame display)

### 3.6 Results Summary (as of Runs 1-5)

| Run | Key Change | Outcome |
|---|---|---|
| 1. Muon baseline | Default optimizer | No learning (Muon fails on 3.6K params) |
| 2. Adam + absolute reward | Fixed optimizer | Lazy agent (Identity 99.5%) |
| 3. MGR + masking | Reward + exploration fix | **Optimal circuit discovered** (F=1.0 in 2 steps) |
| 4. Fixed noise | NISQ-realistic noise | Same circuit, F=0.983 (noise ceiling) |
| 5. Meta-noise | Domain randomization | Same circuit, robust across all noise regimes |

Key insight: MGR + action masking solves non-monotonic fidelity landscapes. The architecture is noise-robust and generalizes across noise distributions without observing the noise parameters.

### 3.7 Research Roadmap

**Phase 2 (next): Multi-target generalization**
- Train a single conditional policy across all 4 Bell states
- Requires adding target state info to the observation (17-dim → 32-dim)
- Tests whether the architecture produces a general circuit compiler, not a memorized solution

**Phase 3 (future): Hardware relevance**
- Hardware-native gate sets (IBM SX/Rz/CX, Google √iSWAP/CZ, IonQ Rxx)
- 3-qubit GHZ state scaling
- Transpiler benchmarking (RL agent vs Qiskit optimization_level=3)

See [EXPERIMENTS.md](../EXPERIMENTS.md) for detailed experiment specs and results.

This manual is comprehensive -- refer back for details as you build and extend the project.
