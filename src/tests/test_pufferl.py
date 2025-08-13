import gymnasium
import pufferlib.emulation
import pufferlib.vector
import numpy as np

class SamplePufferEnv(pufferlib.PufferEnv):
    def __init__(self, foo=0, bar=1, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 2
        super().__init__(buf)
        self.foo = foo
        self.bar = bar

    def reset(self, seed=0):
        self.observations[:] = self.observation_space.sample()  # Fills shared buffer: shape (num_agents, 1) but vectorized will stack
        return self.observations, []

    def step(self, action):
        self.observations[:] = self.observation_space.sample()
        self.rewards[:] = np.random.uniform(0, 1, size=(self.num_agents,))  # Added non-zero rewards for demo
        self.terminals[:] = np.random.choice([True, False], size=(self.num_agents,))
        self.truncations[:] = np.random.choice([True, False], size=(self.num_agents,))
        infos = [{'infos': 'is a list of dictionaries'}] * self.num_agents
        return self.observations, self.rewards, self.terminals, self.truncations, infos

    def close(self):
        pass

if __name__ == '__main__':
    # Serial: Sync, single process
    serial_vecenv = pufferlib.vector.make(
        SamplePufferEnv, num_envs=2, backend=pufferlib.vector.Serial,
        env_kwargs={'bar': 4}  # Pass kwargs to all envs
    )
    observations, infos = serial_vecenv.reset()
    actions = serial_vecenv.action_space.sample()  # Shape: (num_envs * num_agents,)
    o, r, d, t, i = serial_vecenv.step(actions)
    print('Serial VecEnv Outputs:')
    print('Observations shape:', o.shape)  # Expected: (4, 1) -> 2 envs * 2 agents
    print('Observations (random -1 to 1):', o)  # e.g., [[-0.5], [0.3], [0.8], [-0.2]]
    print('Rewards shape:', r.shape)  # (4,)
    print('Rewards (random 0-1):', r)  # e.g., [0.7, 0.2, 0.9, 0.4]
    print('Terminals:', d)  # e.g., [False, True, False, False]
    print('Truncations:', t)  # Similar
    serial_vecenv.close()

    # Multiprocessing: Async, multi-workers
    mp_vecenv = pufferlib.vector.make(
        SamplePufferEnv, num_envs=2, num_workers=2, batch_size=1,  # batch_size=1 means recv small chunks
        backend=pufferlib.vector.Multiprocessing
    )
    mp_vecenv.async_reset()  # Start reset in background
    o, r, d, t, i, env_ids, masks = mp_vecenv.recv()  # Recv batch (size batch_size * num_agents)
    print('\nMultiprocessing VecEnv Outputs (First Batch):')
    print('Observations shape:', o.shape)  # (1*2, 1) = (2,1) since batch_size=1, but from one worker's env
    print('Observations:', o)
    print('Rewards:', r)  # (2,)
    print('Env IDs:', env_ids)  # Which envs this batch is from, e.g., [0]
    print('Masks:', masks)  # (2,) -> 1.0 if active, 0.0 if done

    actions = mp_vecenv.action_space.sample()  # Sample for all, but send only for batch
    mp_vecenv.send(actions[:len(o)])  # Send for the received batch size
    o, r, d, t, i, env_ids, masks = mp_vecenv.recv()  # Next batch ready while others run
    print('Next Batch Observations:', o)
    mp_vecenv.close()