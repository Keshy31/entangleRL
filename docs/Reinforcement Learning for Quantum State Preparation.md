# Reinforcement Learning for Quantum State Preparation in a Simulated Quantum System

## Abstract

This project applies reinforcement learning (RL) to quantum state preparation in a simulated quantum environment. Starting from |00>, an RL agent learns to apply quantum gates to reach a target Bell state with high fidelity. The environment is built using QuTiP for quantum simulations, wrapped in a Gymnasium interface for RL compatibility. Training uses PPO via PufferLib 3.0 for vectorized execution on GPU hardware. A Pygame-based visualization engine provides interactive demos with Bloch spheres and policy insights. No physical quantum hardware is required -- everything runs in simulation. The project emphasizes rich observation spaces (including two-qubit correlators), clean reward signals, and noise robustness, making it ideal for educational purposes.

## Introduction

Quantum state preparation is a fundamental challenge in quantum computing: configuring qubits into specific states needed for algorithms, sensing, or communication. Traditional methods rely on hand-crafted gate sequences or optimization solvers, which can be inefficient or brittle under noise. Reinforcement learning offers a data-driven alternative where an agent experiments with gates, learns from rewards, and develops adaptive strategies.

This project uses PPO, a stable RL algorithm for policy optimization in stochastic environments, accelerated by PufferLib for parallel simulations. The agent learns to prepare entangled Bell pairs, demonstrating RL's potential in quantum control.

Key features:
- Software Simplicity: Python-based simulation with QuTiP, Gymnasium, PufferLib, Pygame, and Matplotlib.
- Educational Focus: Teaches quantum basics (qubits, gates, fidelity, noise) alongside RL concepts (PPO, observation design, reward shaping).
- Rich Observations: 17-dimensional observation space including two-qubit correlators that let the agent "see" entanglement.
- GPU Leverage: Vectorized training with 32+ parallel environments.
- Scalability: Extend to more qubits, custom gates, continuous actions, or noise robustness via meta-RL.

## Glossary

- Qubit: The quantum version of a bit. Can be in superposition (a probabilistic mix of 0 and 1) or entangled with other qubits.
- Quantum State: Describes a qubit system's configuration. For two qubits, it's a 4D complex vector or a 4x4 density matrix.
- Superposition: A qubit in multiple states at once until measured, when it collapses to a definite outcome.
- Entanglement: Qubits correlated beyond classical means. Bell state: measuring one qubit determines the other's outcome.
- Quantum Gate: A reversible operation that transforms states. H creates superposition; CNOT entangles; X flips bits; Z flips phases.
- Density Matrix: Matrix representation of states, useful for mixed/noisy systems. Diagonals are probabilities; off-diagonals are coherences.
- Fidelity: Similarity metric between two states (0 = different, 1 = identical). Our success measure.
- Decoherence/Noise: Environmental interactions that degrade quantum states over time. Simulated via collapse operators in QuTiP.
- Bloch Sphere: 3D visualization of a single qubit's state (North pole = |0>, South = |1>, Equator = superpositions).
- Two-Qubit Correlators: Expectation values like <XX>, <YY>, <ZZ> that capture correlations between qubits. Essential for observing entanglement.
- PPO: Proximal Policy Optimization -- an RL algorithm using neural nets with clipped updates for stability.
- Vectorized Environments: Multiple simulation instances running in parallel for faster data collection.
- Meta-RL: RL that learns to adapt quickly to new conditions (e.g., varying noise levels).
- Episode: One preparation attempt from initial state to target or timeout.

## System Overview

The project is a fully software-based system with four interconnected components:

1. **Quantum Simulation (QuTiP Backend)**: Simulates quantum dynamics classically. Manages qubit states, gate operations, noise models, and fidelity calculations.

2. **RL Environment (QuantumPrepEnv)**: Gymnasium environment wrapping QuTiP. Provides a 17-dimensional observation space (6 single-qubit Pauli expectations + 9 two-qubit correlators + fidelity + step progress), 9 discrete gate actions, and a fidelity-based reward signal.

3. **Training System (PufferLib 3.0 + PPO)**: Vectorizes environments via GymnasiumPufferEnv wrapper. PuffeRL handles the training loop: evaluate() collects rollouts, train() computes PPO updates and returns logs. MLP policy (hidden_size=128) with TensorBoard logging.

4. **Visualization Engine (Pygame)**: Renders Bloch spheres, gate probability bars, and fidelity meters. Loads trained policies for AI-only demos.

Data Flow: Env (QuTiP sim) -> Trainer (PPO updates) -> Saved Model -> Pygame (inference + vis).

## Environment Design

- **Observation Space** (17 floats in [-1, 1]):
  - [0:6] Single-qubit Pauli expectations: <X0>, <Y0>, <Z0>, <X1>, <Y1>, <Z1>
  - [6:15] Two-qubit correlators: <XX>, <XY>, <XZ>, <YX>, <YY>, <YZ>, <ZX>, <ZY>, <ZZ>
  - [15] Current fidelity (squared)
  - [16] Normalized step count (current_step / max_steps)

  The two-qubit correlators are critical: single-qubit expectations alone cannot distinguish the Bell state from a maximally mixed state (both give all zeros for single-qubit Paulis). The Bell state has <XX>=+1, <YY>=-1, <ZZ>=+1.

- **Action Space**: Discrete(9) -- H(Q0), H(Q1), X(Q0), X(Q1), Z(Q0), Z(Q1), CNOT(0->1), CNOT(1->0), Identity.

- **Reward**: `fidelity^2 - 0.01` per step (absolute fidelity, not delta), with +5.0 completion bonus when fidelity > 0.95.

- **Episode Termination**: Fidelity > 0.95 (success) or 50 steps reached (timeout).

- **Noise**: Configurable amplitude damping, dephasing, depolarizing, and bit-flip channels via QuTiP's mesolve. Meta-noise mode randomizes rates each episode.

Example:

    from src.environment.quantum_env import QuantumPrepEnv
    env = QuantumPrepEnv()
    obs, info = env.reset()         # obs.shape == (17,)
    obs, r, term, _, info = env.step(0)  # H on Q0
    obs, r, term, _, info = env.step(6)  # CNOT(0->1) -> Bell state, fidelity ~1.0

## Training Implementation

- **PufferLib Setup**: QuantumPrepEnv wrapped via GymnasiumPufferEnv, vectorized with pufferlib.vector.make() using Multiprocessing backend (32 envs, 4 workers).
- **Trainer**: PuffeRL class from PufferLib 3.0. Loop: evaluate() collects rollouts, train() updates policy and returns combined logs (env stats and loss metrics).
- **Policy**: MLP (pufferlib.models.Default, hidden_size=128). LSTM deferred until MLP proves the environment is learnable.
- **Hyperparameters**: learning_rate=3e-4, gamma=0.99, clip_coef=0.2, update_epochs=4, batch_size=2048, ent_coef=0.01 (linearly decayed to 0.005).
- **Logging**: TensorBoard tracks fidelity, entanglement, action distribution, all PPO losses, and SPS.
- **Output**: Saved model checkpoint (.pth file in models/).

## Visualization and Inference Engine

- Pygame renders Bloch spheres (via QuTiP/Matplotlib images blitted to surface), gate probability bars, and a fidelity meter.
- AI-only mode: loads trained policy, runs inference (argmax actions), renders each step with delays for video recording.
- Loads pufferlib.models.Default with matching hidden_size=128.

## Workflow

1. Install dependencies (requires Linux/WSL for training).
2. Run sanity checks: `python -m src.tests.test_environment`
3. Train: `python -m src.training.train`
4. Monitor: `tensorboard --logdir=logs/tensorboard`
5. Visualize: `python -m src.visualization.engine`
6. Iterate: adjust hyperparameters, add noise, scale up timesteps.

## Educational Value

- Visual demos show the agent learning quantum state preparation in real-time.
- Bloch sphere animations illustrate how gates transform qubit states.
- Two-qubit correlators demonstrate why entanglement requires joint measurements.
- The progression from random exploration to optimal policy demonstrates RL fundamentals.

## Future Extensions

- More qubits (scale to 3-4 for GHZ states)
- Continuous rotation gates (Rx, Ry, Rz with angle parameters)
- Meta-RL for noise robustness across varying decoherence rates
- Integration with Qiskit for hybrid simulated/real quantum backends
- Multi-agent cooperative preparation for distributed quantum systems
