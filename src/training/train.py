import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import pufferlib
import pufferlib.emulation
import pufferlib.pufferl
import pufferlib.vectorization
from torch.utils.tensorboard import SummaryWriter
from src.environment.quantum_env import QuantumPrepEnv

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=64):
        super().__init__()
        input_size = observation_space.shape[0]
        action_size = action_space.n
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
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Steps per rollout (batch = num_envs * rollout_steps = 2048)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Minibatches per update")
    parser.add_argument("--update-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
    parser.add_argument("--torch-deterministic", type=strtobool, default=True)
    parser.add_argument("--cuda", type=strtobool, default=True)
    parser.add_argument("--track", type=strtobool, default=True)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-mode", type=strtobool, default=False, help="Quick test params")
    parser.add_argument("--random-agent", type=strtobool, default=False, help="Use random actions for baseline")

    args = parser.parse_args()

    if args.test_mode:
        args.num_envs = 4
        args.rollout_steps = 32
        args.total_timesteps = 10000
        args.num_minibatches = 4
        args.update_epochs = 4
        print("Test mode enabled: Reduced parameters for quick debug run.")

    args.batch_size = args.num_envs * args.rollout_steps

    run_name = f"QuantumPrep_{args.exp_name}_{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}") if args.track else None
    if writer:
        writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Env Binding
    env_creator = lambda: QuantumPrepEnv(meta_noise=True)  # Align with plan for adaptive noise
    binding = pufferlib.emulation.Binding(env_creator=env_creator, env_name="QuantumPrepEnv")

    # Vectorized Envs
    vecenv = pufferlib.vectorization.Multiprocessing(
        binding=binding,
        num_envs=args.num_envs,
        num_workers=8,  # CPU cores alignment
        device=device,
    )
    vecenv.seed(args.seed)

    # Policy (Custom PyTorch)
    policy = CustomPolicy(binding.observation_space, binding.action_space)

    # Trainer (PuffeRL)
    trainer = pufferlib.pufferl.PuffeRL(
        config=args,  # Pass args as config
        vecenv=vecenv,
        policy=policy,
    )

    # Entropy Decay
    num_updates = args.total_timesteps // args.batch_size
    ent_decay_step = (0.001 - args.ent_coef) / num_updates

    # Training Loop
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        trainer.evaluate()  # Collect interactions
        trainer.train()  # Update on batch
        trainer.mean_and_log()  # Log aggregation

        # Manual Aggregation (if needed, as mean_and_log may handle)
        avg_fid, avg_len, win_rate = aggregate_metrics(trainer.infos)  # Assume trainer has infos

        # Log Custom
        if writer:
            writer.add_scalar("losses/value_loss", trainer.value_loss, global_step)
            writer.add_scalar("losses/policy_loss", trainer.policy_loss, global_step)
            writer.add_scalar("losses/entropy", trainer.entropy_loss, global_step)
            writer.add_scalar("losses/kl", trainer.kl, global_step)
            writer.add_scalar("charts/avg_fidelity", avg_fid, global_step)
            writer.add_scalar("charts/avg_episode_length", avg_len, global_step)
            writer.add_scalar("charts/win_rate", win_rate, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        global_step += args.batch_size

        # Entropy Decay
        trainer.ent_coef = max(trainer.ent_coef + ent_decay_step, 0.001)

        # Save
        if update % 10 == 0:
            trainer.save_checkpoint(f"models/{run_name}_update{update}.pt")

    # Final Save/Cleanup
    trainer.save_checkpoint(f"models/{run_name}.pt")
    print(f"Model saved to models/{run_name}.pt")
    trainer.close()
    if writer:
        writer.close()

if __name__ == "__main__":
    main()