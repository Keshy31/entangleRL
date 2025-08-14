import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib.pufferl import PuffeRL
import torch

from src.environment.quantum_prep_env import QuantumPrepEnv

def env_creator():
    env = QuantumPrepEnv(
        num_qubits=2,
        # Add params: gate_set=['H', 'X', 'Z', 'CNOT'], noise_level=0.0, etc.
    )
    return pufferlib.emulation.GymnasiumPufferEnv(env)

vecenv = pufferlib.vector.make(
    env_creator,
    num_workers=4,
    envs_per_worker=32,  # Total 128 envs
)

policy = pufferlib.models.MLP(
    input_size=vecenv.single_observation_space.shape[0],
    output_size=vecenv.single_action_space.n,
    hidden_size=64,
    hidden_layers=2,
)

config = pufferlib.pufferl.Config(
    learning_rate=3e-4,
    gamma=0.99,
    clip_ratio=0.2,
    ppo_epochs=4,
    update_batch_size=2048,
    entropy_coef=0.01,
    total_timesteps=1_000_000,  # Adjust as needed
)

trainer = PuffeRL(config, vecenv, policy)

trainer.train()

scores = trainer.evaluate(num_episodes=100)
print(f"Mean fidelity: {sum(scores) / len(scores):.4f}")

torch.save(trainer.agent.state_dict(), 'ppo_quantum.pth')

trainer.close()
vecenv.close()