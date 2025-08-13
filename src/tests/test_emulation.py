# src/tests/test_emulation.py (renamed or repurposed as single env test)
import numpy as np
from src.environment.quantum_env import QuantumPrepEnv  # Adjust import path as needed

if __name__ == '__main__':
    env = QuantumPrepEnv()
    obs, info = env.reset()
    print('Observation space:', env.single_observation_space)  # Box for single-agent
    print('Action space:', env.single_action_space)  # Discrete(9)
    print('Initial obs:', obs)  # Shape (1, 6)
    action = np.array([0])  # Action as array for PufferEnv compatibility
    o, r, d, t, i = env.step(action)
    print('Next obs:', o)
    print('Reward:', r)
    env.close()