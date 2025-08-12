# EntangleRL: Reinforcement Learning for Quantum State Preparation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An educational project that applies reinforcement learning to quantum state preparation in a simulated quantum environment. Watch an AI learn to create entangled quantum states like Bell pairs through trial and error, complete with beautiful visualizations!

## 🎯 Project Overview

This project uses PPO (Proximal Policy Optimization) to train an AI agent that learns to prepare quantum states by applying sequences of quantum gates. Starting from a basic initial state (all qubits in |0⟩), the agent learns to reach target states like entangled Bell states with high fidelity.

### Key Features

- **🎮 No Quantum Hardware Required**: Pure software simulation using QuTiP
- **⚡ GPU Accelerated**: Fast training with PufferLib on RTX 4080
- **🎨 Beautiful Visualizations**: Animated Bloch spheres and real-time policy insights
- **📚 Educational Focus**: Perfect for learning both quantum computing and RL concepts
- **🔧 Modular Design**: Easy to extend with more qubits, gates, or noise models

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/entangleRL.git
cd entangleRL

# Install dependencies
pip install -r requirements.txt

# Test the quantum environment
python scripts/test_environment.py

# Train the agent (takes ~1 hour on RTX 4080)
python scripts/train_agent.py

# Watch the trained agent in action
python scripts/run_demo.py
```

## 📁 Project Structure

```
entangleRL/
├── docs/                           # Documentation and guides
├── src/                           # Main source code
│   ├── environment/               # Quantum RL environment
│   ├── training/                  # PPO training system  
│   ├── visualization/             # Pygame rendering engine
│   └── utils/                     # Shared utilities
├── scripts/                       # Executable scripts
├── tests/                         # Unit tests
├── models/                        # Saved trained models
├── logs/                          # Training logs and metrics
└── assets/                        # Images and demo materials
```

## 🧠 How It Works

1. **Quantum Simulation**: QuTiP simulates quantum states and gate operations
2. **RL Environment**: Gymnasium wrapper makes it compatible with RL algorithms
3. **PPO Training**: PufferLib provides vectorized, GPU-accelerated training
4. **Visualization**: Pygame renders beautiful Bloch spheres and training insights

## 📖 Learn More

- [📋 Complete Project Whitepaper](docs/Reinforcement%20Learning%20for%20Quantum%20State%20Preparation.md)
- [📚 Comprehensive Guidebook](docs/Guidebook%20for%20Reinforcement%20Learning%20in%20Quantum%20State%20Preparation.md)

## 🎥 Demo Videos

*Coming soon - watch AI learn quantum entanglement!*

## 🔬 Research Context

This project aligns with 2025 trends in quantum machine learning, particularly:
- RL for quantum error correction
- Adaptive quantum control systems  
- Educational tools for quantum computing

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- QuTiP team for quantum simulation tools
- PufferLib for high-performance RL training
- OpenAI for PPO algorithm development
