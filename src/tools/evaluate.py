"""
Evaluate a trained QuantumPrepEnv policy checkpoint.

Loads either a wrapped checkpoint ({"model_state_dict", "config"}, as saved by
src.training.train) or a flat state dict (.pth), rolls out episodes with the
greedy policy (argmax) by default, prints the discovered circuit per episode,
and reports aggregate statistics. Optionally dumps full per-step trajectories
(gate probabilities, value estimates, fidelities, density matrices) to JSON
for visualization -- this is the data source for the planned demo video.

Usage:
    python -m src.tools.evaluate --checkpoint models/fixed_noise/final.pt
    python -m src.tools.evaluate --checkpoint models/mlp_100k_mgr_masking.pth --episodes 5
    python -m src.tools.evaluate --checkpoint models/noiseless_mgr/checkpoint_26624.pt --json rollout.json
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import pufferlib.emulation
import pufferlib.models

from src.environment.quantum_env import QuantumPrepEnv

NOISE_KEYS = ("amplitude_damping_rate", "dephasing_rate", "depolarizing_rate",
              "bit_flip_rate", "thermal_occupation")


def load_checkpoint(path):
    """Return (state_dict, config) from a wrapped checkpoint or a flat state dict."""
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"], ckpt.get("config", {})
    return ckpt, {}


def dm_to_json(state):
    """Density matrix -> nested [re, im] lists (JSON-serializable)."""
    arr = state.full()
    return [[[round(c.real, 6), round(c.imag, 6)] for c in row] for row in arr]


def rollout_episode(env, policy, sample, rng):
    raw = env.env  # underlying QuantumPrepEnv
    obs, info = env.reset()
    target_name = raw.target_names[raw.target_index] if raw.multi_target else None
    steps = [{
        "step": 0,
        "action": None,
        "gate": None,
        "reward": 0.0,
        "fidelity": float(info["fidelity"]),
        "max_fidelity": float(info["max_fidelity"]),
        "entanglement": float(info["entanglement"]),
        "density_matrix": dm_to_json(raw.current_state),
    }]

    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = policy(obs_t)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = F.softmax(logits.squeeze(0), dim=-1).numpy()
        if sample:
            action = int(rng.choice(len(probs), p=probs / probs.sum()))
        else:
            action = int(np.argmax(probs))

        obs, reward, terminated, truncated, info = env.step(action)
        steps.append({
            "step": int(info["steps"]),
            "action": action,
            "gate": raw._gate_name_map[action],
            "gate_probs": [round(float(p), 4) for p in probs],
            "value_estimate": round(float(value.squeeze()), 4),
            "reward": round(float(reward), 4),
            "fidelity": float(info["fidelity"]),
            "max_fidelity": float(info["max_fidelity"]),
            "entanglement": float(info["entanglement"]),
            "density_matrix": dm_to_json(raw.current_state),
        })

    return {
        "target": target_name,
        "steps": steps,
        "length": int(info["steps"]),
        "final_fidelity": float(info["fidelity"]),
        "completed": bool(terminated),
        "episode_return": float(info["episode_return"]),
        "noise": {k: float(getattr(raw, k)) for k in NOISE_KEYS},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained QuantumPrepEnv policy checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt (wrapped) or .pth (flat state dict)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true",
                        help="Sample actions from the policy instead of greedy argmax")
    parser.add_argument("--noiseless", action="store_true",
                        help="Force all noise channels off regardless of checkpoint config")
    parser.add_argument("--target", type=str, default=None,
                        choices=list(QuantumPrepEnv.TARGET_NAMES),
                        help="Pin every episode to one Bell target (multi-target checkpoints)")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Policy hidden size (only used for flat checkpoints without a config)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override the environment's max episode length")
    parser.add_argument("--json", type=str, default=None,
                        help="Dump full per-step trajectories to this JSON file")
    args = parser.parse_args()

    state_dict, config = load_checkpoint(args.checkpoint)

    env_kwargs = dict(config.get("env", {}))
    if args.noiseless:
        env_kwargs.update({k: 0.0 for k in NOISE_KEYS})
        env_kwargs["meta_noise"] = False
    if args.max_steps is not None:
        env_kwargs["max_steps"] = args.max_steps
    if args.target is not None:
        env_kwargs["multi_target"] = True
        env_kwargs["fixed_target_index"] = list(QuantumPrepEnv.TARGET_NAMES).index(args.target)

    hidden_size = config.get("train", {}).get("hidden_size", args.hidden_size)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    raw_env = QuantumPrepEnv(**env_kwargs)
    env = pufferlib.emulation.GymnasiumPufferEnv(raw_env)
    policy = pufferlib.models.Default(env=env, hidden_size=hidden_size)
    policy.load_state_dict(state_dict)
    policy.eval()

    mode = "sampled" if args.sample else "greedy"
    src = config.get("experiment_name", "none (flat state dict, env defaults)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config:     {src}")
    target_desc = args.target if args.target else (
        "random per episode" if env_kwargs.get("multi_target") else "single (|Phi+>)")
    print(f"Policy:     {mode} | episodes: {args.episodes} | "
          f"noise: {'forced off' if args.noiseless else 'from config'} | "
          f"target: {target_desc}")
    print("-" * 78)

    episodes = []
    for ep in range(args.episodes):
        result = rollout_episode(env, policy, args.sample, rng)
        episodes.append(result)
        gates = [s["gate"] for s in result["steps"] if s["gate"] is not None]
        circuit = " -> ".join(gates) if gates else "(no actions)"
        if len(circuit) > 46:
            circuit = circuit[:43] + "..."
        status = "completed" if result["completed"] else "truncated"
        tgt = f"[{result['target']}] " if result["target"] else ""
        print(f"Ep {ep + 1:>3}: F_end={result['final_fidelity']:.4f}  {status:9s}  "
              f"len={result['length']:>2}  ret={result['episode_return']:+.2f}  {tgt}{circuit}")

    f_end = np.array([e["final_fidelity"] for e in episodes])
    lengths = np.array([e["length"] for e in episodes])
    completed = np.array([float(e["completed"]) for e in episodes])
    returns = np.array([e["episode_return"] for e in episodes])
    print("-" * 78)
    print(f"Completion: {completed.mean() * 100:.0f}%   "
          f"F_end: {f_end.mean():.4f} +/- {f_end.std():.4f}   "
          f"len: {lengths.mean():.1f}   ret: {returns.mean():+.2f}")

    circuits = Counter(
        (e["target"], tuple(s["gate"] for s in e["steps"] if s["gate"] is not None))
        for e in episodes
    )
    print("Circuits:")
    for (target, circuit), count in sorted(
            circuits.items(), key=lambda kv: (kv[0][0] or "", -kv[1])):
        name = " -> ".join(circuit) if circuit else "(no actions)"
        if target:
            name = f"[{target}] {name}"
        if len(name) > 70:
            name = name[:67] + "..."
        print(f"  {count:>3}x  {name}")

    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "config": config,
            "env_kwargs": env_kwargs,
            "mode": mode,
            "seed": args.seed,
            "gate_names": raw_env._gate_name_map,
            "episodes": episodes,
        }
        if raw_env.multi_target:
            payload["target_names"] = list(raw_env.target_names)
        with open(args.json, "w") as f:
            json.dump(payload, f)
        print(f"\nWrote trajectories to {args.json}")

    env.close()


if __name__ == "__main__":
    main()
