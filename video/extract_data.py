"""
Extract all data needed by the demo-video manim scenes into video/data/.

Runs rollouts through src.tools.evaluate (subprocess, same CLI documented in
EXPERIMENTS.md) and dumps TensorBoard scalar curves with EventAccumulator.
Must run in the training environment (WSL venv), from the repo root:

    python video/extract_data.py

The manim scenes (video/scenes/) only read the JSONs produced here, so they
can render from any environment without qutip/pufferlib installed.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "video" / "data"

# (output name, checkpoint, episodes, sampled?)  -- seed fixed for reproducibility
ROLLOUTS = [
    ("lazy_agent", "models/mlp_100k_adam_baseline.pth", 3, False),
    ("ckpt_06k", "models/noiseless_mgr/checkpoint_6144.pt", 8, True),
    ("ckpt_18k", "models/noiseless_mgr/checkpoint_18432.pt", 8, True),
    ("final", "models/noiseless_mgr/final.pt", 3, False),
    ("fixed_noise", "models/fixed_noise/final.pt", 3, False),
    ("meta_noise", "models/meta_noise/final.pt", 8, False),
]

# run -> {alias: scalar tag}
CURVES = {
    "noiseless_mgr": {
        "final_fidelity": "Env/final_fidelity",
        "episode_completed": "Env/episode_completed",
        "episode_length": "Env/episode_length",
        "entropy": "Loss/entropy",
    },
    "mlp_100k_adam_baseline": {
        "entropy": "Loss/entropy",
        "identity_frac": "Actions/Action_8",
        "fidelity": "Env/Fidelity",
    },
}


def extract_rollouts():
    for name, checkpoint, episodes, sampled in ROLLOUTS:
        out = DATA / f"{name}.json"
        cmd = [
            sys.executable, "-m", "src.tools.evaluate",
            "--checkpoint", checkpoint,
            "--episodes", str(episodes),
            "--seed", "42",
            "--json", str(out),
        ]
        if sampled:
            cmd.append("--sample")
        print(f"\n=== {name}: {' '.join(cmd[2:])}")
        subprocess.run(cmd, cwd=REPO, check=True)


def extract_curves():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    payload = {}
    for run, tags in CURVES.items():
        acc = EventAccumulator(str(REPO / "logs" / "tensorboard" / run),
                               size_guidance={"scalars": 0})
        acc.Reload()
        payload[run] = {}
        for alias, tag in tags.items():
            events = acc.Scalars(tag)
            payload[run][alias] = {
                "steps": [e.step for e in events],
                "values": [round(e.value, 5) for e in events],
            }
            print(f"{run}/{alias}: {len(events)} points")
    with open(DATA / "curves.json", "w") as f:
        json.dump(payload, f)
    print(f"Wrote {DATA / 'curves.json'}")


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    extract_rollouts()
    extract_curves()
    print("\nDone.")
