"""
Configurable PPO training for QuantumPrepEnv via PufferLib.

Usage:
    python -m src.training.train                          # default (noiseless MGR)
    python -m src.training.train --experiment fixed_noise  # named preset
    python -m src.training.train --experiment_name my_run --amplitude_damping_rate 0.05
"""
import argparse
import json
import os
import sys
import time

import pufferlib
import pufferlib.vector
import pufferlib.models
from pufferlib.pufferl import PuffeRL
from src.environment.quantum_env import QuantumPrepEnv
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

# ---------------------------------------------------------------------------
# Experiment presets
# ---------------------------------------------------------------------------
EXPERIMENT_PRESETS = {
    "noiseless_mgr": {
        "experiment_name": "noiseless_mgr",
        "env": {},
        "train": {"total_timesteps": 100_000},
    },
    "fixed_noise": {
        # Measured noise ceiling: F=0.9834 for optimal 2-gate path.
        # Threshold 0.93 gives ~5% margin below ceiling.
        "experiment_name": "fixed_noise",
        "env": {
            "amplitude_damping_rate": 0.05,
            "dephasing_rate": 0.02,
            "depolarizing_rate": 0.01,
            "gate_time": 0.1,
            "completion_threshold": 0.93,
        },
        "train": {"total_timesteps": 200_000},
    },
    "meta_noise": {
        "experiment_name": "meta_noise",
        "env": {
            "meta_noise": True,
            "completion_threshold": 0.80,
        },
        "train": {"total_timesteps": 500_000},
    },
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_ENV_KWARGS = {
    "meta_noise": False,
    "amplitude_damping_rate": 0.0,
    "dephasing_rate": 0.0,
    "depolarizing_rate": 0.0,
    "bit_flip_rate": 0.0,
    "thermal_occupation": 0.0,
    "gate_time": 0.1,
    "completion_threshold": 0.95,
}

DEFAULT_TRAIN_KWARGS = {
    "total_timesteps": 100_000,
    "learning_rate": 3e-4,
    "update_epochs": 8,
    "gamma": 0.95,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "vf_clip_coef": 0.2,
    "batch_size": 2048,
    "minibatch_size": 512,
    "seed": 42,
    "hidden_size": 128,
    "num_envs": 32,
    "num_workers": 4,
    "checkpoint_interval": 25_000,
}

GATE_NAMES = {
    0: "H_Q0", 1: "H_Q1", 2: "X_Q0", 3: "X_Q1",
    4: "Z_Q0", 5: "Z_Q1", 6: "CNOT_01", 7: "CNOT_10", 8: "Identity",
}


def build_config(args):
    """Merge preset -> defaults -> CLI overrides into a final config."""
    env_kwargs = dict(DEFAULT_ENV_KWARGS)
    train_kwargs = dict(DEFAULT_TRAIN_KWARGS)
    experiment_name = "experiment"

    if args.experiment and args.experiment in EXPERIMENT_PRESETS:
        preset = EXPERIMENT_PRESETS[args.experiment]
        experiment_name = preset["experiment_name"]
        env_kwargs.update(preset.get("env", {}))
        train_kwargs.update(preset.get("train", {}))

    # CLI overrides for env kwargs
    for key in DEFAULT_ENV_KWARGS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            env_kwargs[key] = cli_val

    # CLI overrides for train kwargs
    for key in DEFAULT_TRAIN_KWARGS:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            train_kwargs[key] = cli_val

    if args.experiment_name:
        experiment_name = args.experiment_name

    return experiment_name, env_kwargs, train_kwargs


def make_env_creator(env_kwargs):
    """Return a closure that PufferLib can call to create envs in workers."""
    def env_creator(**kwargs):
        env = QuantumPrepEnv(**env_kwargs)
        return pufferlib.emulation.GymnasiumPufferEnv(env, **kwargs)
    return env_creator


def log_hparams(writer, experiment_name, env_kwargs, train_kwargs):
    """Write hyperparameters to TensorBoard's hparams plugin."""
    hparam_dict = {"experiment": experiment_name}
    for k, v in env_kwargs.items():
        hparam_dict[f"env/{k}"] = v
    for k, v in train_kwargs.items():
        hparam_dict[f"train/{k}"] = v

    metric_dict = {
        "hparam/final_fidelity": 0.0,
        "hparam/final_completion_rate": 0.0,
    }
    # Use the low-level hparams API to avoid TensorBoard auto-closing the writer
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def save_checkpoint(policy, config, path):
    """Save model weights and experiment config together."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "config": config,
    }, path)


def train(experiment_name, env_kwargs, train_kwargs):
    """Main training loop."""
    # --- Paths ---
    tb_dir = f"logs/tensorboard/{experiment_name}"
    model_dir = f"models/{experiment_name}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    full_config = {
        "experiment_name": experiment_name,
        "env": env_kwargs,
        "train": train_kwargs,
    }

    # Save config to disk
    config_path = os.path.join(model_dir, "config.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)

    # --- PufferLib config ---
    # load_config parses sys.argv internally; isolate it from our CLI args
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    args = pufferlib.pufferl.load_config('default')
    sys.argv = saved_argv

    args['train']['env'] = 'quantum_prep'
    args['train']['use_rnn'] = False
    args['train']['seed'] = train_kwargs['seed']
    args['train']['batch_size'] = train_kwargs['batch_size']
    args['train']['bptt_horizon'] = 'auto'
    args['train']['minibatch_size'] = train_kwargs['minibatch_size']
    args['train']['max_minibatch_size'] = 2048
    args['train']['total_timesteps'] = train_kwargs['total_timesteps']
    args['train']['learning_rate'] = train_kwargs['learning_rate']
    args['train']['update_epochs'] = train_kwargs['update_epochs']
    args['train']['gamma'] = train_kwargs['gamma']
    args['train']['gae_lambda'] = train_kwargs['gae_lambda']
    args['train']['clip_coef'] = train_kwargs['clip_coef']
    args['train']['ent_coef'] = train_kwargs['ent_coef']
    args['train']['vf_coef'] = train_kwargs['vf_coef']
    args['train']['vf_clip_coef'] = train_kwargs['vf_clip_coef']
    args['train']['optimizer'] = 'adam'
    args['train']['anneal_lr'] = False

    args['vec']['backend'] = 'Multiprocessing'
    args['vec']['num_envs'] = train_kwargs['num_envs']
    args['vec']['num_workers'] = train_kwargs['num_workers']
    args['vec']['batch_size'] = max(1, train_kwargs['num_envs'] // train_kwargs['num_workers'])

    # --- Vectorized Environment ---
    env_creator = make_env_creator(env_kwargs)
    vecenv = pufferlib.vector.make(
        env_creator,
        backend=args['vec']['backend'],
        num_workers=args['vec']['num_workers'],
        num_envs=args['vec']['num_envs'],
        batch_size=args['vec']['batch_size'],
    )

    # --- Policy ---
    policy = pufferlib.models.Default(
        env=vecenv,
        hidden_size=train_kwargs['hidden_size'],
    ).cuda()

    # --- Trainer ---
    trainer = PuffeRL(args['train'], vecenv, policy)
    writer = SummaryWriter(log_dir=tb_dir)
    log_hparams(writer, experiment_name, env_kwargs, train_kwargs)

    total_timesteps = train_kwargs['total_timesteps']
    checkpoint_interval = train_kwargs['checkpoint_interval']
    last_checkpoint_step = 0
    last_log_time = time.time()

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  TensorBoard: {tb_dir}")
    print(f"  Checkpoints: {model_dir}")
    print(f"  Total timesteps: {total_timesteps:,}")
    noise_active = any(env_kwargs.get(k, 0) > 0 for k in
                       ("amplitude_damping_rate", "dephasing_rate",
                        "depolarizing_rate", "bit_flip_rate")) or env_kwargs.get("meta_noise", False)
    print(f"  Noise: {'ON' if noise_active else 'OFF'}")
    if noise_active:
        for k in ("amplitude_damping_rate", "dephasing_rate", "depolarizing_rate",
                   "bit_flip_rate", "thermal_occupation", "gate_time"):
            print(f"    {k}: {env_kwargs.get(k, 0)}")
    print(f"  Completion threshold: {env_kwargs['completion_threshold']}")
    print(f"{'=' * 60}\n")

    update_count = 0
    while trainer.global_step < total_timesteps:
        trainer.evaluate()
        logs = trainer.train()

        if logs is None:
            continue

        update_count += 1
        step = trainer.global_step

        # --- PPO losses ---
        for key in ('policy_loss', 'value_loss', 'entropy', 'approx_kl',
                     'clipfrac', 'explained_variance'):
            full_key = f'losses/{key}'
            if full_key in logs:
                writer.add_scalar(f'Loss/{key}', logs[full_key], step)

        if 'SPS' in logs:
            writer.add_scalar('Performance/SPS', logs['SPS'], step)

        # --- Environment info metrics (dynamic discovery) ---
        env_metrics = {}
        for key, val in logs.items():
            if key.startswith('environment/'):
                metric_name = key.split('/', 1)[1]
                env_metrics[metric_name] = val
                # Group action metrics under Actions/, everything else under Env/
                if metric_name.startswith('action_'):
                    tb_key = f'Actions/{GATE_NAMES.get(int(metric_name.split("_")[1]), metric_name)}'
                else:
                    tb_key = f'Env/{metric_name}'
                writer.add_scalar(tb_key, val, step)

        # --- Console progress (every ~10 seconds) ---
        now = time.time()
        if now - last_log_time > 10.0:
            fid = env_metrics.get('fidelity', 0)
            max_fid = env_metrics.get('max_fidelity', 0)
            completion = env_metrics.get('completed', 0)
            ep_ret = env_metrics.get('episode_return', 0)
            ep_len = env_metrics.get('steps', 0)
            sps = logs.get('SPS', 0)
            entropy = logs.get('losses/entropy', 0)
            pct = step / total_timesteps * 100
            print(f"  [{pct:5.1f}%] step={step:>7,}  fid={fid:.3f}  "
                  f"max_fid={max_fid:.3f}  done%={completion:.2f}  "
                  f"ret={ep_ret:+.2f}  len={ep_len:.1f}  "
                  f"ent={entropy:.3f}  SPS={sps:.0f}")
            last_log_time = now

        # --- Periodic checkpoint ---
        if step - last_checkpoint_step >= checkpoint_interval:
            ckpt_path = os.path.join(model_dir, f"checkpoint_{step}.pt")
            save_checkpoint(trainer.policy, full_config, ckpt_path)
            last_checkpoint_step = step

    # --- Final save ---
    final_path = os.path.join(model_dir, "final.pt")
    save_checkpoint(trainer.policy, full_config, final_path)
    # Also save as flat .pth for backward compatibility with visualization engine
    torch.save(trainer.policy.state_dict(), f"models/{experiment_name}.pth")

    # Update hparam metrics with final values
    writer.add_scalar("hparam/final_fidelity",
                       env_metrics.get('fidelity', 0), step)
    writer.add_scalar("hparam/final_completion_rate",
                       env_metrics.get('completed', 0), step)

    print(f"\n{'=' * 60}")
    print(f"  Training complete: {step:,} steps")
    print(f"  Final model: {final_path}")
    print(f"  TensorBoard: tensorboard --logdir={tb_dir}")
    print(f"{'=' * 60}")

    trainer.close()
    vecenv.close()
    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train QuantumPrepEnv with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Experiment selection
    parser.add_argument("--experiment", type=str, default=None,
                        choices=list(EXPERIMENT_PRESETS.keys()),
                        help="Load a named experiment preset")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Custom name (overrides preset name)")

    # Environment overrides
    for key, default in DEFAULT_ENV_KWARGS.items():
        arg_type = type(default) if not isinstance(default, bool) else None
        if isinstance(default, bool):
            parser.add_argument(f"--{key}", action="store_true", default=None)
        else:
            parser.add_argument(f"--{key}", type=arg_type, default=None)

    # Training overrides
    for key, default in DEFAULT_TRAIN_KWARGS.items():
        parser.add_argument(f"--{key}", type=type(default), default=None)

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    name, env_kw, train_kw = build_config(cli_args)
    train(name, env_kw, train_kw)
