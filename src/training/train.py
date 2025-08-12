import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import pufferlib
import pufferlib.emulation
import pufferlib.models
import pufferlib.vectorization
from torch.utils.tensorboard import SummaryWriter

from src.environment.quantum_env import QuantumPrepEnv


def strtobool(val):
    """Converts a string to a boolean value."""
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val}")


def main():
    """
    Main function to run the training script.
    
    This script sets up the environment, agent, and training loop,
    then starts the training process and saves the trained model.
    """
    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser(description="PPO Training for Quantum State Preparation")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num-envs", type=int, default=128, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=2048, help="Number of steps per environment per policy rollout")
    parser.add_argument("--total-timesteps", type=int, default=10_000_000, help="Total timesteps to train for")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--num-minibatches", type=int, default=32, help="Number of minibatches to split a batch into")
    parser.add_argument("--update-epochs", type=int, default=10, help="Number of epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Maximum gradient norm")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, `torch.backends.cudnn.deterministic=True`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True, help="if toggled, this experiment will be tracked with TensorBoard")
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"), help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # --- 2. Experiment Setup ---
    run_name = f"QuantumPrep_{args.exp_name}_{args.seed}_{int(time.time())}"
    if args.track:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
    else:
        writer = None

    # --- 3. PufferLib Environment Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    if args.torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Define the environment creation function
    env_creator = lambda: QuantumPrepEnv()

    # Create the PufferLib environment binding
    binding = pufferlib.emulation.Binding(
        env_creator=env_creator,
        env_name="QuantumPrepEnv",
        suppress_env_prints=False,
    )

    # --- 4. Define the Policy ---
    class Policy(pufferlib.models.Policy):
        def __init__(self, binding, *args, **kwargs):
            super().__init__(binding)
            self.encoder = pufferlib.pytorch.ReluOpen(
                input_size=binding.env_observation_space.shape[0],
                hidden_size=64,
            )
            self.critic_net = nn.Linear(64, 1)
            self.actor = nn.Linear(64, binding.env_action_space.n)

        def critic(self, hidden):
            return self.critic_net(hidden)

        def encode_observations(self, env_outputs):
            return self.encoder(env_outputs), None

        def decode_actions(self, hidden, lookup):
            # The second return value is the value prediction
            return self.actor(hidden), self.critic(hidden)

    # --- 5. Instantiate Agent and Optimizer ---
    agent = pufferlib.frameworks.cleanrl.make_agent(
        Policy,
        binding=binding,
        args=vars(args)
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # --- 6. PufferLib Vectorized Environment ---
    envs = pufferlib.vectorization.Serial(
        binding,
        num_envs=args.num_envs,
        device=device,
    )
    
    # --- 7. Training Loop ---
    obs, _ = envs.reset()
    global_step = 0
    start_time = time.time()
    
    # Storage for rollouts
    num_steps = args.num_steps
    obs_storage = torch.zeros(num_steps, args.num_envs, *binding.env_observation_space.shape).to(device)
    action_storage = torch.zeros(num_steps, args.num_envs, *binding.env_action_space.shape).to(device)
    logprob_storage = torch.zeros(num_steps, args.num_envs).to(device)
    reward_storage = torch.zeros(num_steps, args.num_envs).to(device)
    done_storage = torch.zeros(num_steps, args.num_envs).to(device)
    value_storage = torch.zeros(num_steps, args.num_envs).to(device)

    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # --- Rollout Phase ---
        for step in range(num_steps):
            global_step += 1 * args.num_envs
            obs_storage[step] = obs
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs)
                value_storage[step] = value.flatten()

            action_storage[step] = action
            logprob_storage[step] = logprob

            obs, reward, terminated, truncated, info = envs.step(action)
            done = torch.logical_or(terminated, truncated)
            reward_storage[step] = reward.view(-1)
            done_storage[step] = done
            
            if writer and "fidelity" in info:
                for i, fid in enumerate(info["fidelity"]):
                    if info["done"][i]:
                        writer.add_scalar("charts/episodic_fidelity", fid, global_step)
                        writer.add_scalar("charts/episodic_length", info["steps"][i], global_step)

        # --- GAE and Value Target Calculation ---
        with torch.no_grad():
            next_value = agent.get_value(obs).reshape(1, -1)
            advantages = torch.zeros_like(reward_storage).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_storage[t + 1]
                    nextvalues = value_storage[t + 1]
                delta = reward_storage[t] + args.gamma * nextvalues * nextnonterminal - value_storage[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + value_storage

        # --- Update Phase ---
        b_obs = obs_storage.reshape((-1,) + binding.env_observation_space.shape)
        b_logprobs = logprob_storage.reshape(-1)
        b_actions = action_storage.reshape((-1,) + binding.env_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = value_storage.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                
                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # --- Logging ---
        if writer:
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # --- 8. Save Model and Cleanup ---
    model_path = f"models/{run_name}.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    envs.close()
    if writer:
        writer.close()

if __name__ == "__main__":
    main()
