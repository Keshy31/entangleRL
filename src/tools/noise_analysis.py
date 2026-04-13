"""
Noise ceiling analysis for QuantumPrepEnv.

Runs the optimal gate sequence (H -> CNOT) under specified noise parameters
and reports the achievable fidelity. Use this before training to determine
an appropriate completion_threshold for noisy experiments.

Usage:
    python -m src.tools.noise_analysis
    python -m src.tools.noise_analysis --amplitude_damping_rate 0.1 --dephasing_rate 0.05
"""
import argparse
import numpy as np
from src.environment.quantum_env import QuantumPrepEnv


OPTIMAL_BELL_PATH = [0, 6]  # H(Q0) -> CNOT(0->1)
GATE_NAMES = {0: "H(Q0)", 1: "H(Q1)", 6: "CNOT(0->1)", 7: "CNOT(1->0)", 8: "Identity"}


def analyze_noise_ceiling(env_kwargs, gate_sequence=None, num_trials=100):
    """Run a fixed gate sequence under noise and report fidelity statistics.

    Because Lindblad evolution via mesolve is deterministic for a given state
    and collapse operators, a single trial suffices for fixed noise. Multiple
    trials are useful for meta_noise mode where rates are randomized.
    """
    if gate_sequence is None:
        gate_sequence = OPTIMAL_BELL_PATH

    fidelities = []
    for _ in range(num_trials):
        env = QuantumPrepEnv(**env_kwargs)
        obs, info = env.reset()
        for action in gate_sequence:
            obs, reward, terminated, truncated, info = env.step(action)
        fidelities.append(info['fidelity'])
        env.close()

    return np.array(fidelities)


def main():
    parser = argparse.ArgumentParser(description="Noise ceiling analysis")
    parser.add_argument("--amplitude_damping_rate", type=float, default=0.05)
    parser.add_argument("--dephasing_rate", type=float, default=0.02)
    parser.add_argument("--depolarizing_rate", type=float, default=0.01)
    parser.add_argument("--bit_flip_rate", type=float, default=0.0)
    parser.add_argument("--thermal_occupation", type=float, default=0.0)
    parser.add_argument("--gate_time", type=float, default=0.1)
    parser.add_argument("--meta_noise", action="store_true")
    parser.add_argument("--num_trials", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10,
                        help="Test circuits up to this many gates to check for compensating sequences")
    args = parser.parse_args()

    env_kwargs = {
        "amplitude_damping_rate": args.amplitude_damping_rate,
        "dephasing_rate": args.dephasing_rate,
        "depolarizing_rate": args.depolarizing_rate,
        "bit_flip_rate": args.bit_flip_rate,
        "thermal_occupation": args.thermal_occupation,
        "gate_time": args.gate_time,
        "meta_noise": args.meta_noise,
        "completion_threshold": 1.0,  # never terminate early
    }

    print("=" * 60)
    print("NOISE CEILING ANALYSIS")
    print("=" * 60)
    print(f"\nNoise parameters:")
    for k in ("amplitude_damping_rate", "dephasing_rate", "depolarizing_rate",
              "bit_flip_rate", "thermal_occupation", "gate_time"):
        print(f"  {k:30s} = {env_kwargs[k]}")
    print(f"  {'meta_noise':30s} = {args.meta_noise}")

    # 1. Optimal 2-gate path
    print(f"\n--- Optimal path: H(Q0) -> CNOT(0->1) ---")
    fids = analyze_noise_ceiling(env_kwargs, OPTIMAL_BELL_PATH, args.num_trials)
    print(f"  Fidelity:  mean={fids.mean():.6f}  std={fids.std():.6f}  "
          f"min={fids.min():.6f}  max={fids.max():.6f}")

    optimal_ceiling = fids.mean()

    # 2. Alternative path: H(Q1) -> CNOT(1->0)
    alt_path = [1, 7]
    print(f"\n--- Alternative path: H(Q1) -> CNOT(1->0) ---")
    fids_alt = analyze_noise_ceiling(env_kwargs, alt_path, args.num_trials)
    print(f"  Fidelity:  mean={fids_alt.mean():.6f}  std={fids_alt.std():.6f}  "
          f"min={fids_alt.min():.6f}  max={fids_alt.max():.6f}")

    # 3. Noiseless baseline (for reference)
    noiseless_kwargs = {**env_kwargs, "amplitude_damping_rate": 0.0, "dephasing_rate": 0.0,
                        "depolarizing_rate": 0.0, "bit_flip_rate": 0.0, "thermal_occupation": 0.0}
    print(f"\n--- Noiseless baseline ---")
    fids_clean = analyze_noise_ceiling(noiseless_kwargs, OPTIMAL_BELL_PATH, 1)
    print(f"  Fidelity:  {fids_clean[0]:.6f}")

    # 4. Depth scan: does adding more gates help?
    print(f"\n--- Depth scan (identity padding after optimal path) ---")
    all_actions = list(range(9))
    for extra_depth in range(1, args.max_depth - 1):
        best_fid = 0.0
        best_seq = None
        # Try each non-repeating extension
        for action in all_actions:
            seq = OPTIMAL_BELL_PATH + [action] * extra_depth
            fids_d = analyze_noise_ceiling(env_kwargs, seq, 1)
            if fids_d[0] > best_fid:
                best_fid = fids_d[0]
                best_seq = seq
        gate_str = " -> ".join(GATE_NAMES.get(a, f"Gate{a}") for a in best_seq)
        marker = " <-- BETTER" if best_fid > optimal_ceiling + 1e-6 else ""
        print(f"  Depth {len(best_seq)}: best F={best_fid:.6f}  ({gate_str}){marker}")

    # 5. Recommendation
    print(f"\n{'=' * 60}")
    print(f"RECOMMENDATION")
    print(f"{'=' * 60}")
    noise_ceiling = optimal_ceiling
    suggested_threshold = round(noise_ceiling - 0.05, 2)
    suggested_threshold = max(suggested_threshold, 0.5)  # floor
    print(f"  Noise ceiling (2-gate optimal): {noise_ceiling:.4f}")
    print(f"  Fidelity loss from noise:       {1.0 - noise_ceiling:.4f}")
    print(f"  Suggested completion_threshold:  {suggested_threshold:.2f}")
    if noise_ceiling < 0.95:
        print(f"  WARNING: Noise ceiling ({noise_ceiling:.4f}) is BELOW default "
              f"threshold (0.95).")
        print(f"  The agent will NEVER receive the +5.0 completion bonus with "
              f"the default threshold.")
        print(f"  You MUST lower completion_threshold for this noise regime.")
    else:
        print(f"  Default threshold (0.95) is reachable under this noise level.")


if __name__ == "__main__":
    main()
