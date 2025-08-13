import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pufferlib.vector
from pufferlib import pufferl
from src.environment.quantum_env import QuantumPrepEnv  # Adjust path

class CustomPolicy(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_size = env.single_observation_space.shape[0]
        action_size = env.single_action_space.n
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, action_size)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        hidden = self.encoder(x)
        return self.actor(hidden), self.critic(hidden)

def aggregate_metrics(infos):
    fids = []
    lengths = []
    wins = 0
    for inf in infos:
        if inf:
            fid = inf.get('fidelity', 0)
            fids.append(fid)
            lengths.append(inf.get('steps', 0))
            if fid > 0.95:
                wins += 1
    avg_fid = np.mean(fids) if fids else 0
    avg_len = np.mean(lengths) if lengths else 0
    win_rate = wins / len(fids) if fids else 0
    return avg_fid, avg_len, win_rate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PPO Training for Quantum State Preparation")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--cuda", action="store_true", help="If toggled, cuda will be enabled")
    parser.add_argument("--track", action="store_true", help="if toggled, this experiment will be tracked with TensorBoard")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with reduced parameters")
    args = parser.parse_args()

    # Base config (adapted from project defaults)
    config = {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'clip_ratio': 0.2,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'update_epochs': 4,
        'num_minibatches': 4,
        'total_timesteps': 1_000_000,
        'num_envs': 128,
        'bptt_horizon': 128,  # Steps per rollout
        'num_workers': 8,
        'max_grad_norm': 0.5,
        'torch_deterministic': True,
        'cuda': args.cuda,
        'seed': args.seed,
        'track': args.track,
    }

    if args.test_mode:
        print("Test mode enabled: Reduced parameters for quick debug run.")
        config['num_envs'] = 8
        config['bptt_horizon'] = 32
        config['total_timesteps'] = 10000
        config['num_workers'] = 2
        config['update_epochs'] = 2
        config['num_minibatches'] = 2

    # Set up experiment tracking
    run_name = f"QuantumPrep_PPO_{config['seed']}_{int(time.time())}"
    log_dir = f"runs/{run_name}"
    writer = SummaryWriter(log_dir) if config['track'] else None
    if writer:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])))

    # Set device and seed
    device = 'cuda' if torch.cuda.is_available() and config['cuda'] else 'cpu'
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if config['torch_deterministic']:
        torch.backends.cudnn.deterministic = True

    # Vectorized Envs (directly with PufferEnv class)
    vecenv = pufferlib.vector.make(
        QuantumPrepEnv,
        num_envs=config['num_envs'],
        num_workers=config['num_workers'],
        batch_size=config['num_envs'],  # Full batch
        backend=pufferlib.vector.Multiprocessing,
        env_kwargs={'meta_noise': True}  # Enable meta-RL
    )

    # Policy
    policy = CustomPolicy(vecenv.driver_env).to(device)

    # Trainer
    trainer = pufferl.PuffeRL(config, vecenv, policy)

    # Training Loop
    num_updates = config['total_timesteps'] // (config['num_envs'] * config['bptt_horizon'])
    global_step = 0
    start_time = time.time()
    for update in range(1, num_updates + 1):
        trainer.evaluate()  # Optional eval
        logs = trainer.train()  # Main training step

        # Custom metrics from infos
        avg_fid, avg_len, win_rate = aggregate_metrics(trainer.infos)

        # Log
        if writer:
            writer.add_scalar("losses/value_loss", logs.get('value_loss', 0), global_step)
            writer.add_scalar("losses/policy_loss", logs.get('policy_loss', 0), global_step)
            writer.add_scalar("losses/entropy", logs.get('entropy_loss', 0), global_step)
            writer.add_scalar("losses/approx_kl", logs.get('approx_kl', 0), global_step)
            writer.add_scalar("charts/avg_fidelity", avg_fid, global_step)
            writer.add_scalar("charts/avg_episode_length", avg_len, global_step)
            writer.add_scalar("charts/win_rate", win_rate, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        global_step += config['num_envs'] * config['bptt_horizon']

        # Checkpoint
        if update % max(1, num_updates // 10) == 0:
            torch.save(policy.state_dict(), f"models/{run_name}_update{update}.pt")

    # Final save and cleanup
    torch.save(policy.state_dict(), f"models/{run_name}.pt")
    print(f"Model saved to models/{run_name}.pt")
    if writer:
        writer.close()
    trainer.close()