# src/tests/test_vectorization.py
import pufferlib.vector
from src.environment.quantum_env import QuantumPrepEnv  # Adjust import path as needed

if __name__ == '__main__':
    # Serial backend test
    print("=== Serial Backend Test ===")
    vecenv = pufferlib.vector.make(QuantumPrepEnv, num_envs=2, backend=pufferlib.vector.Serial)
    observations, infos = vecenv.reset()
    print('Observations:', observations)  # Expect shape (2, 6) with initial Pauli values
    actions = vecenv.action_space.sample()  # Shape (2,)
    o, r, d, t, i = vecenv.step(actions)
    print('Rewards:', r)  # Non-zero based on fidelity change
    vecenv.close()

    # Multiprocessing backend test
    print("\n=== Multiprocessing Backend Test ===")
    mp_vecenv = pufferlib.vector.make(
        QuantumPrepEnv,
        num_envs=4,  # Example: 4 envs for divisibility
        num_workers=2,  # 2 workers, each handling 2 envs
        batch_size=2,  # Batch size matching per-worker envs for simplicity
        backend=pufferlib.vector.Multiprocessing
    )
    mp_vecenv.async_reset()
    o, r, d, t, i, env_ids, masks = mp_vecenv.recv()
    print('MP Observations:', o)  # Expect shape (2, 6) per batch
    print('MP Rewards:', r)
    print('Env IDs:', env_ids)
    print('Masks:', masks)
    actions = mp_vecenv.action_space.sample()[:len(o)]  # Actions for the batch
    mp_vecenv.send(actions)
    # Optional: Receive next batch to verify
    o, r, d, t, i, env_ids, masks = mp_vecenv.recv()
    print('Next MP Observations:', o)
    print('Next MP Rewards:', r)
    print('Next MP Env IDs:', env_ids)
    print('Next MP Masks:', masks)


    actions = mp_vecenv.action_space.sample()[:len(o)]  # Actions for the batch
    mp_vecenv.send(actions)
    # Optional: Receive next batch to verify
    o, r, d, t, i, env_ids, masks = mp_vecenv.recv()
    print('Next MP Observations:', o)
    print('Next MP Rewards:', r)
    print('Next MP Env IDs:', env_ids)
    print('Next MP Masks:', masks)
    mp_vecenv.close()