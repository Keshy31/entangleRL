# EntangleRL: Reinforcement Learning for Quantum State Preparation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An RL agent discovers optimal quantum circuits from scratch -- no hardcoded solutions, no domain-specific heuristics. Starting from random exploration, PPO + a custom Moving-Goalpost Reward learns the **optimal 2-gate Bell state circuit** (H then CNOT) in under 40K timesteps, robust across noiseless, fixed-noise, and domain-randomized noise conditions.

## Project Overview

This project uses PPO (Proximal Policy Optimization) via PufferLib 3.0 to train an agent that prepares quantum states by applying sequences of quantum gates. Starting from |00>, the agent learns to reach the Bell state with high fidelity.

### Key Features

- **No Quantum Hardware Required** -- pure software simulation using QuTiP
- **GPU Accelerated** -- vectorized training with PufferLib on CUDA
- **Rich Observations** -- 17-dimensional observation space with single-qubit Paulis, two-qubit correlators, fidelity, and step progress
- **Moving-Goalpost Reward** -- rewards only new fidelity records per episode, preventing lazy-agent traps in non-monotonic quantum landscapes
- **Dynamic Action Masking** -- blocks immediate repetition of self-inverse gates, forcing exploration
- **Modular Design** -- extend with more qubits, gates, or noise models

## Prerequisites

### Why Linux / WSL?

PufferLib's vectorized training backend uses Python's `multiprocessing` with the `fork` start method, which is only available on Linux and macOS. **Windows does not support `fork()`**, so training will fail on native Windows. If you're on Windows, you must run everything inside WSL (Windows Subsystem for Linux).

### Windows Setup (WSL)

If you're on Windows, install WSL first -- this gives you a full Linux environment inside Windows:

```powershell
# Run in PowerShell (as Administrator)
wsl --install           # installs Ubuntu by default
# Restart your PC when prompted, then open the "Ubuntu" app to finish setup
```

Once inside WSL (the Ubuntu terminal), install Python and venv support:

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git
```

> **Tip**: Your Windows files are accessible at `/mnt/c/Users/<YourName>/` inside WSL, but for best performance clone the repo into the native Linux filesystem (e.g. `~/projects/`).

### Linux / macOS

No extra setup needed -- just make sure you have Python 3.12+ and `pip` installed.

## Quick Start

All commands below should be run in a **Linux terminal** (native Linux, macOS, or WSL on Windows).

```bash
# Clone the repo
git clone https://github.com/yourusername/entangleRL.git
cd entangleRL

# Create and activate a virtual environment
python3 -m venv .entangleRL-env
source .entangleRL-env/bin/activate   # always activate before working on the project

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Sanity-check the environment
python -m src.tests.test_environment

# Train the agent (default: noiseless MGR)
python -m src.training.train --experiment noiseless_mgr

# Or run a specific experiment preset
python -m src.training.train --experiment fixed_noise
python -m src.training.train --experiment meta_noise

# Analyze noise ceiling before a noisy experiment
python -m src.tools.noise_analysis

# Visualize a trained policy (requires a display -- see note below)
python -m src.visualization.engine
```

> **Virtual environment reminder**: Every time you open a new terminal, re-activate the venv before running any project commands:
> ```bash
> cd entangleRL
> source .entangleRL-env/bin/activate
> ```

> **WSL display note**: The Pygame visualization requires a graphical display. If you're in WSL, you need either WSLg (Windows 11, enabled by default) or an X server like VcXsrv on Windows 10. TensorBoard and headless training work without any display.

## Project Structure

```
entangleRL/
├── docs/                           # Whitepaper and guidebook
├── src/
│   ├── environment/
│   │   └── quantum_env.py          # QuantumPrepEnv (Gymnasium)
│   ├── training/
│   │   └── train.py                # Configurable experiment runner (PPO via PufferLib)
│   ├── tools/
│   │   └── noise_analysis.py       # Pre-experiment noise ceiling analysis
│   ├── visualization/
│   │   └── engine.py               # Pygame inference + Bloch sphere rendering
│   └── tests/
│       ├── test_environment.py     # Environment sanity checks (incl. noise tests)
│       ├── test_emulation.py       # PufferLib emulation smoke test
│       ├── test_vectorization.py   # Vector env backend tests
│       └── test_pufferl.py         # PufferLib API demos
├── models/                         # Saved model checkpoints + configs
├── logs/                           # TensorBoard training logs
├── EXPERIMENTS.md                  # Detailed experiment log and roadmap
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
| Reward | Moving-Goalpost: `max(0, F_t - F_max)` for new records, `-0.01` otherwise, +5.0 bonus at F > 0.95 |
| Action Masking | Previous action blocked each step (all gates are self-inverse) |
| Termination | Fidelity > 0.95 or 50 steps reached |
| Target state | Bell state (|00> + |11>)/sqrt(2) |

## Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam | PufferLib 3.0 defaults to Muon, which fails on tiny (3.6K param) networks |
| Learning rate | 3e-4 | Standard for Adam PPO |
| Entropy coefficient | 0.05 (constant) | High floor prevents premature collapse in deceptive landscapes |
| Discount (gamma) | 0.95 | Short horizon suits the max-50-step episodes |
| Update epochs | 8 | Deeper optimization per batch to exploit rare high-reward discoveries |
| LR annealing | Disabled | Short runs don't benefit from cosine decay |

## Reward Design: Why Moving-Goalpost?

The initial state |00> has intrinsic fidelity 0.5 with the Bell state target. A naive absolute reward (`F^2 - penalty`) creates a **lazy-agent trap**: the agent learns that doing nothing yields reward 0.24/step, while the optimal first gate (Hadamard) *drops* fidelity to 0.25 before the entangling CNOT gate raises it to 1.0.

The Moving-Goalpost Reward (MGR) solves this by only rewarding new fidelity records within each episode. The fidelity dip to 0.25 yields zero reward (not positive or deeply negative), while the jump from 0.5 to 1.0 produces a large positive signal. Combined with dynamic action masking (preventing gate self-cancellation), this drives the agent to discover the non-obvious H→CNOT sequence.

See the [research analysis](docs/Guidebook%20for%20Reinforcement%20Learning%20in%20Quantum%20State%20Preparation.md) for the full mathematical treatment.

## Training Results

### Phase 1: Architecture Validation (complete)

| Run | Key Change | Fidelity | Outcome |
|-----|-----------|----------|---------|
| 1. Muon baseline | Default optimizer | 0.267 | No learning (Muon fails on 3.6K params) |
| 2. Adam + absolute reward | Fixed optimizer | 0.500 | Lazy agent (Identity 99.5%) |
| **3. MGR + masking** | **Moving-Goalpost reward** | **1.000** | **Optimal circuit in 2 steps** |
| 4. Fixed noise | NISQ-realistic noise | 0.983 | Same circuit, matches noise ceiling |
| 5. Meta-noise | Domain randomization | 0.918-1.0* | Robust across all noise regimes |

\*Range reflects noise variation across episodes. The agent discovers the same H→CNOT circuit regardless of noise.

### Key Insight: Non-Monotonic Fidelity Landscapes

The initial state |00⟩ has intrinsic fidelity 0.5 with the Bell target. The optimal path requires *reducing* fidelity to 0.25 (via Hadamard) before the entangling CNOT raises it to 1.0. Absolute rewards trap the agent at the 0.5 plateau. MGR solves this by only rewarding new fidelity records, making the dip cost-neutral.

### Next: Multi-Target Generalization

Training a single conditional policy to prepare all 4 Bell states, testing whether the architecture produces a general circuit compiler rather than a memorized solution. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full roadmap.

TensorBoard logs for all runs are under `logs/tensorboard/`. Launch with:
```bash
tensorboard --logdir=logs/tensorboard
```

## Learn More

- [Project Whitepaper](docs/Reinforcement%20Learning%20for%20Quantum%20State%20Preparation.md)
- [Comprehensive Guidebook](docs/Guidebook%20for%20Reinforcement%20Learning%20in%20Quantum%20State%20Preparation.md)

## Research Roadmap

**Phase 1** (complete): Architecture validation -- MGR + masking solves deceptive landscapes, robust under noise

**Phase 2** (next): Multi-target generalization -- single policy across all 4 Bell states, conditional on target observation

**Phase 3** (future): Hardware relevance -- native gate sets (IBM/Google/IonQ), transpiler benchmarking, 3-qubit GHZ scaling

## Research Context

This project explores the intersection of quantum computing and reinforcement learning:
- Reward shaping for non-monotonic optimization landscapes (Moving-Goalpost Reward)
- Noise-robust circuit discovery via domain randomization
- RL as a general-purpose quantum circuit compiler
- Adaptive quantum control systems

## License

This project is licensed under the MIT License.

## Acknowledgments

- QuTiP team for quantum simulation tools
- PufferLib for high-performance RL training
- OpenAI for PPO algorithm development
