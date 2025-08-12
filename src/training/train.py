import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pufferlib
import pufferlib.emulation
import pufferlib.clean_ppo
import pufferlib.vectorization
from torch.utils.tensorboard import SummaryWriter
from src.environment.quantum_env import QuantumPrepEnv  # Adjust path if needed

def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")

class Policy(nn.Module):
    def __init__(self, input_size, action_size, hidden_size=64):
        super().__init__()
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

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Quantum State Preparation")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--rollout-steps", type=int, default=16, help="Steps per rollout (batch_size = num_envs * rollout_steps)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--num-minibatches", type=int, default=4, help="Minibatches per update")
    parser.add_argument("--update-epochs", type=int, default=4, help="Epochs per update")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Initial entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--test-mode", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="Override for quick test runs")

    args = parser.parse_args()

    if args.test_mode:
        args.num_envs = 4
        args.rollout_steps = 32
        args.total_timesteps = 10000
        args.num_minibatches = 4
        args.update_epochs = 4
        print("Test mode enabled: Reduced parameters for quick run.")

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
    env_creator = lambda: QuantumPrepEnv(meta_noise=True)  # Enable meta-RL
    binding = pufferlib.emulation.Binding(
        env_creator=env_creator,
        env_name="QuantumPrepEnv",
    )

    # Vectorized Envs (Multiprocessing for parallel)
    envs = pufferlib.vectorization.Multiprocessing(
        binding,
        num_envs=args.num_envs,
        num_workers=8,  # Adjust based on CPU cores
        device=device,
    )

    # Policy and Trainer
    policy = Policy(input_size=binding.observation_space.shape[0], action_size=binding.action_space.n)
    trainer = pufferlib.clean_ppo.Trainer(
        policy=policy,
        envs=envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        device=device,
    )

    # Entropy Decay Scheduler (linear to 0.001)
    num_updates = args.total_timesteps // args.batch_size
    ent_decay = (0.001 - args.ent_coef) / num_updates  # Per update decrement

    # Training Loop
    global_step = 0
    start_time = time.time()
    fid_history = []  # For avg fid
    ep_lengths = []
    wins = 0

    for update in range(1, num_updates + 1):
        # Train one update (rollout + compute + optimize)
        metrics = trainer.train(steps=args.batch_size)  # Handles rollout/GAE/updates

        # Aggregate Quantum Metrics from Metrics/Infos
        if 'info' in metrics:
            for info_batch in metrics['info']:  # List of dicts
                for inf in info_batch:
                    if 'done' in inf and inf['done']:
                        fid = inf['fidelity']
                        fid_history.append(fid)
                        ep_lengths.append(inf['steps'])
                        if fid > 0.95:
                            wins += 1

        # Log
        if writer:
            writer.add_scalar("losses/value_loss", metrics['value_loss'], global_step)
            writer.add_scalar("losses/policy_loss", metrics['policy_loss'], global_step)
            writer.add_scalar("losses/entropy", metrics['entropy'], global_step)
            writer.add_scalar("losses/kl", metrics['kl'], global_step)
            if fid_history:
                writer.add_scalar("charts/avg_fidelity", np.mean(fid_history), global_step)
                writer.add_scalar("charts/avg_episode_length", np.mean(ep_lengths), global_step)
                writer.add_scalar("charts/win_rate", wins / len(fid_history), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        global_step += args.batch_size

        # Entropy Decay
        trainer.ent_coef += ent_decay
        trainer.ent_coef = max(trainer.ent_coef, 0.001)  # Clamp

        # Periodic Save
        if update % 10 == 0:
            torch.save(policy.state_dict(), f"models/{run_name}_update{update}.pth")

    # Final Save and Cleanup
    torch.save(policy.state_dict(), f"models/{run_name}.pth")
    print(f"Model saved to models/{run_name}.pth")
    envs.close()
    if writer:
        writer.close()

if __name__ == "__main__":
    main()