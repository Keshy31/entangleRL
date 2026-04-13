# EntangleRL: Reinforcement Learning for Quantum State Preparation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational project that applies reinforcement learning to quantum state preparation in a simulated quantum environment. An AI agent learns to create entangled quantum states like Bell pairs through trial and error.

## Project Overview

This project uses PPO (Proximal Policy Optimization) via PufferLib 3.0 to train an agent that prepares quantum states by applying sequences of quantum gates. Starting from |00>, the agent learns to reach the Bell state with high fidelity.

### Key Features

- **No Quantum Hardware Required** -- pure software simulation using QuTiP
- **GPU Accelerated** -- vectorized training with PufferLib on CUDA
- **Rich Observations** -- 17-dimensional observation space with single-qubit Paulis, two-qubit correlators, fidelity, and step progress
- **Educational Focus** -- learn quantum computing and RL concepts together
- **Modular Design** -- extend with more qubits, gates, or noise models

## Quick Start

Training requires Linux (or WSL) due to PufferLib's multiprocessing backend using `fork()`.

```bash
# Clone and install
git clone https://github.com/yourusername/entangleRL.git
cd entangleRL
python -m venv .entangleRL-env
source .entangleRL-env/bin/activate
pip install -r requirements.txt
pip install -e .

# Sanity-check the environment
python -m src.tests.test_environment

# Train the agent
python -m src.training.train

# Visualize a trained policy
python -m src.visualization.engine
```

## Project Structure

```
entangleRL/
├── docs/                           # Whitepaper and guidebook
├── src/
│   ├── environment/
│   │   └── quantum_env.py          # QuantumPrepEnv (Gymnasium)
│   ├── training/
│   │   └── train.py                # PufferLib PPO training loop
│   ├── visualization/
│   │   └── engine.py               # Pygame inference + Bloch sphere rendering
│   └── tests/
│       ├── test_environment.py     # Environment sanity checks
│       ├── test_emulation.py       # PufferLib emulation smoke test
│       ├── test_vectorization.py   # Vector env backend tests
│       └── test_pufferl.py         # PufferLib API demos
├── models/                         # Saved model checkpoints (.pth)
├── logs/                           # TensorBoard training logs
├── requirements.txt
├── setup.py
└── README.md
```

## How It Works

1. **Quantum Simulation**: QuTiP simulates 2-qubit density matrices and gate operations
2. **RL Environment**: `QuantumPrepEnv` wraps QuTiP in a Gymnasium interface with a 17-dim observation (6 single-qubit Paulis + 9 two-qubit correlators + fidelity + step progress) and 9 discrete gate actions
3. **PPO Training**: PufferLib vectorizes 32 parallel environments and trains with PPO (MLP policy, `hidden_size=128`)
4. **Visualization**: Pygame renders Bloch spheres, gate probability bars, and a fidelity meter

## Environment Details

| Property | Value |
|---|---|
| Observation | 17 floats: `<X0>,<Y0>,<Z0>,<X1>,<Y1>,<Z1>`, `<XX>,...,<ZZ>`, fidelity, step/max |
| Actions | 9 discrete: H(Q0), H(Q1), X(Q0), X(Q1), Z(Q0), Z(Q1), CNOT(0->1), CNOT(1->0), Identity |
| Reward | `fidelity^2 - 0.01` per step, +5.0 bonus at fidelity > 0.95 |
| Termination | Fidelity > 0.95 or 50 steps reached |
| Target state | Bell state (|00> + |11>)/sqrt(2) |

## Learn More

- [Project Whitepaper](docs/Reinforcement%20Learning%20for%20Quantum%20State%20Preparation.md)
- [Comprehensive Guidebook](docs/Guidebook%20for%20Reinforcement%20Learning%20in%20Quantum%20State%20Preparation.md)

## Research Context

This project explores the intersection of quantum computing and reinforcement learning:
- RL for quantum error correction
- Adaptive quantum control systems
- Educational tools for quantum computing

## License

This project is licensed under the MIT License.

## Acknowledgments

- QuTiP team for quantum simulation tools
- PufferLib for high-performance RL training
- OpenAI for PPO algorithm development
