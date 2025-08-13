import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pufferlib
import pufferlib.emulation
import pufferlib.vector
import sys

from pufferlib.vector import make
from pufferlib import pufferl
from torch.utils.tensorboard import SummaryWriter
from src.environment import QuantumPrepEnv

def make_env(meta_noise=True, **kwargs):
    """
    Utility function for creating a GymnasiumPufferEnv
    """
    env = QuantumPrepEnv(meta_noise=meta_noise)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, **kwargs)

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"'{val}' is not a valid boolean value")

class CustomPolicy(nn.Module):
    def __init__(self, env, hidden_size=64):
        super().__init__()
        input_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)

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

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Quantum State Preparation")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--cuda", action="store_true", help="If toggled, cuda will be enabled")
    parser.add_argument("--track", action="store_true", help="if toggled, this experiment will be tracked with TensorBoard")
    parser.add_argument("--test-mode", action="store_true", help="Enable test mode with reduced parameters")
    args = parser.parse_args()

    # Load PufferLib's default config and update it with our args
    config = pufferl.load_config('default')['train']
    config.seed = args.seed
    config.cuda = args.cuda
    config.track = args.track

    if args.test_mode:
        print("Test mode enabled: Reduced parameters for quick debug run.")
        config.num_envs = 8
        config.bptt_horizon = 32
        config.total_timesteps = 10000
        config.num_minibatches = 4
        config.num_workers = 4
        config.update_epochs = 4
        config.exp_name = "test_run"

    # Set up experiment tracking
    run_name = f"QuantumPrep_{config.exp_name}_{config.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}") if config.track else None
    if writer:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])))

    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() and config.cuda else "cpu")
    config.device = device
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if config.torch_deterministic:
        torch.backends.cudnn.deterministic = True

    # Env creator
    env_creator = make_env
    env_kwargs = {'meta_noise': True}

    # Vectorized Envs
    vecenv = make(
        env_creator=make_env,
        num_envs=config.num_envs,
        backend='Multiprocessing',
        num_workers=config.num_workers,
        env_kwargs=env_kwargs,
    )

    # Policy
    policy = CustomPolicy(vecenv.driver_env)

    # Trainer (PuffeRL) - Pass the policy as the 'agent'
    trainer = pufferl.PuffeRL(
        config=config,
        agent=policy,
        vecenv=vecenv,
    )

    # Training Loop
    num_updates = config.total_timesteps // (config.num_envs * config.bptt_horizon)
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        trainer.evaluate()
        trainer.train()
        trainer.mean_and_log()

        # Custom Aggregation
        avg_fid, avg_len, win_rate = aggregate_metrics(trainer.infos)

        # Log Custom to TensorBoard
        if writer:
            writer.add_scalar("losses/value_loss", getattr(trainer, 'value_loss', 0), global_step)
            writer.add_scalar("losses/policy_loss", getattr(trainer, 'policy_loss', 0), global_step)
            writer.add_scalar("losses/entropy", getattr(trainer, 'entropy_loss', 0), global_step)
            writer.add_scalar("losses/kl", getattr(trainer, 'kl', 0), global_step)
            writer.add_scalar("charts/avg_fidelity", avg_fid, global_step)
            writer.add_scalar("charts/avg_episode_length", avg_len, global_step)
            writer.add_scalar("charts/win_rate", win_rate, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        global_step += (config.num_envs * config.bptt_horizon)

        # Save
        if update % int(num_updates / 10) == 0:
            trainer.save_checkpoint(f"models/{run_name}_update{update}.pt")

    # Final Save/Cleanup
    trainer.save_checkpoint(f"models/{run_name}.pt")
    print(f"Model saved to models/{run_name}.pt")
    trainer.close()
    if writer:
        writer.close()

if __name__ == "__main__":
    main()