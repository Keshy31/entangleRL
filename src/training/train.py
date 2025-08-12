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
from src.environment.quantum_env import QuantumPrepEnv

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

def aggregate_infos(infos):
    fids = []
    lengths = []
    wins = 0
    for info_batch in infos:  # List of dicts per env
        if isinstance(info_batch, dict):  # Single
            if 'fidelity' in info_batch:
                fids.append(info_batch['fidelity'])
                lengths.append(info_batch['steps'])
                if info_batch['fidelity'] > 0.95:
                    wins += 1
        else:  # Batched list
            for inf in info_batch:
                if inf and 'fidelity' in inf:
                    fids.append(inf['fidelity'])
                    lengths.append(inf['steps'])
                    if inf['fidelity'] > 0.95:
                        wins += 1
    return np.mean(fids) if fids else 0, np.mean(lengths) if lengths else 0, wins / len(fids) if fids else 0

def main():
    parser = argparse.ArgumentParser(description="PPO Training for Quantum State Preparation")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-steps", type=int, default=16)  # batch=128*16=2048
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
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
        print("Test mode: Reduced params for debug.")

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
    env_creator = lambda: QuantumPrepEnv(meta_noise=True)
    binding = pufferlib.emulation.Binding(env_creator=env_creator, env_name="QuantumPrepEnv")

    # Vectorized Envs
    envs = pufferlib.vectorization.Multiprocessing(
        binding,
        num_envs=args.num_envs,
        num_workers=8,
        device=device,
    )
    envs.seed(args.seed)

    # Policy
    policy = Policy(input_size=binding.observation_space.shape[0], action_size=binding.action_space.n)

    # Trainer
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
        postprocess_callback=aggregate_infos if not args.random_agent else None,
    )

    # Entropy Decay
    num_updates = args.total_timesteps // args.batch_size
    ent_decay_step = (0.001 - args.ent_coef) / num_updates

    # Training Loop
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
        if args.random_agent:
            # Random baseline: Sample actions
            metrics = trainer.train(steps=args.batch_size, random_actions=True)
        else:
            metrics = trainer.train(steps=args.batch_size)

        avg_fid, avg_len, win_rate = metrics.get('custom_metrics', (0, 0, 0))  # From callback

        if writer:
            writer.add_scalar("losses/value_loss", metrics['value_loss'], global_step)
            writer.add_scalar("losses/policy_loss", metrics['policy_loss'], global_step)
            writer.add_scalar("losses/entropy", metrics['entropy'], global_step)
            writer.add_scalar("losses/kl", metrics['kl'], global_step)
            writer.add_scalar("charts/avg_fidelity", avg_fid, global_step)
            writer.add_scalar("charts/avg_episode_length", avg_len, global_step)
            writer.add_scalar("charts/win_rate", win_rate, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        global_step += args.batch_size

        # Decay Entropy
        trainer.ent_coef = max(trainer.ent_coef + ent_decay_step, 0.001)

        # Save
        if update % 10 == 0:
            torch.save(policy.state_dict(), f"models/{run_name}_update{update}.pth")

    # Final Save/Cleanup
    torch.save(policy.state_dict(), f"models/{run_name}.pth")
    print(f"Model saved to models/{run_name}.pth")
    envs.close()
    if writer:
        writer.close()

if __name__ == "__main__":
    main()