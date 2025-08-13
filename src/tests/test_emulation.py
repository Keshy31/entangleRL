import pufferlib.emulation
from src.environment import QuantumPrepEnv  # Assuming your env file

env = pufferlib.emulation.GymnasiumPufferEnv(QuantumPrepEnv(meta_noise=True))
obs, info = env.reset()
print("Observation space:", env.observation_space)  # Should be Box(low, high, shape=(input_size,))
print("Action space:", env.action_space)  # Discrete(n_gates)
print("Initial obs:", obs)  # Flat array from your partial obs (e.g., expectation values)

action = env.action_space.sample()
next_obs, reward, done, truncated, info = env.step(0)
print("After step - Reward:", reward, "Fidelity:", info.get('fidelity', 0), "Done:", done)
next_obs, reward, done, truncated, info = env.step(6)
print("After step - Reward:", reward, "Fidelity:", info.get('fidelity', 0), "Done:", done)

env.close()