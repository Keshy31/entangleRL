import pufferlib.vector
from src.environment import QuantumPrepEnv

env = pufferlib.emulation.GymnasiumPufferEnv(QuantumPrepEnv(meta_noise=True))
vecenv = pufferlib.vector.make(env, num_envs=2, backend=pufferlib.vector.Serial)
observations, infos = vecenv.reset()
actions =vecenv.action_space.sample()

print('Actions:', actions)

# Step 1
observations, rewards, truncations, terminations, infos = vecenv.step(actions)
print('Observations:', observations)
print('Rewards:', rewards)
print('Truncations:', truncations)
print('Terminations:', terminations)
print('Infos:', infos)

# Step 2
observations, rewards, truncations, terminations, infos = vecenv.step(actions)
print('Observations:', observations)
print('Rewards:', rewards)
print('Truncations:', truncations)
print('Terminations:', terminations)
print('Infos:', infos)

vecenv.close()