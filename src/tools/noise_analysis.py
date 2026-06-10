"""
Noise ceiling analysis for QuantumPrepEnv.

Single-target mode (default): runs the optimal gate sequence (H -> CNOT) under
specified noise parameters and reports the achievable fidelity. Use this before
training to determine an appropriate completion_threshold for noisy experiments.

Multi-Bell mode (--multi_bell): for each of the 4 Bell targets, brute-forces the
minimal-depth noiseless-optimal circuits, then ranks every gate ordering under
worst-case noise (all meta-noise rates maxed) and mean meta-noise rates. Reports
per-target noise ceilings, whether gate ordering matters under decoherence, and
a Monte Carlo completion-rate estimate at the proposed threshold under
domain-randomized noise. Use before multi-target noisy experiments (Run 7).

Usage:
    python -m src.tools.noise_analysis
    python -m src.tools.noise_analysis --amplitude_damping_rate 0.1 --dephasing_rate 0.05
    python -m src.tools.noise_analysis --multi_bell
    python -m src.tools.noise_analysis --multi_bell --threshold 0.80 --mc_trials 200
"""
import argparse
import itertools

import numpy as np
from src.environment.quantum_env import QuantumPrepEnv


OPTIMAL_BELL_PATH = [0, 6]  # H(Q0) -> CNOT(0->1)
GATE_NAMES = {0: "H(Q0)", 1: "H(Q1)", 2: "X(Q0)", 3: "X(Q1)", 4: "Z(Q0)",
              5: "Z(Q1)", 6: "CNOT(0->1)", 7: "CNOT(1->0)", 8: "Identity"}

# meta_noise sampling ranges (must match QuantumPrepEnv.reset): U(0, max) per episode
META_NOISE_MAX = {
    "amplitude_damping_rate": 0.2,
    "dephasing_rate": 0.1,
    "depolarizing_rate": 0.05,
    "bit_flip_rate": 0.05,
    "thermal_occupation": 0.1,
}
META_NOISE_MEAN = {k: v / 2 for k, v in META_NOISE_MAX.items()}


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


def run_sequence(env, actions):
    """Apply a gate sequence from reset; return (final_fidelity, max_fidelity)."""
    _, info = env.reset()
    for action in actions:
        _, _, _, _, info = env.step(action)
    return info["fidelity"], info["max_fidelity"]


def find_minimal_circuits(target_index, gate_time, max_depth=5):
    """Brute-force the minimal depth and all noiseless-optimal circuits for a Bell target.

    Enumerates sequences over the 8 non-identity gates with no immediate
    repeats (the env masks repeats, and a repeated gate is never minimal).
    """
    env = QuantumPrepEnv(multi_target=True, fixed_target_index=target_index,
                         gate_time=gate_time, completion_threshold=1.0)
    try:
        for depth in range(1, max_depth + 1):
            optimal = []
            for seq in itertools.product(range(8), repeat=depth):
                if any(seq[i] == seq[i + 1] for i in range(depth - 1)):
                    continue
                f_end, _ = run_sequence(env, seq)
                if f_end > 0.999:
                    optimal.append(list(seq))
            if optimal:
                return depth, optimal
    finally:
        env.close()
    return max_depth, []


def rank_orderings(target_index, circuits, noise_rates, gate_time):
    """Evaluate circuits under fixed noise rates; return [(f_end, seq)] best-first."""
    env = QuantumPrepEnv(multi_target=True, fixed_target_index=target_index,
                         gate_time=gate_time, completion_threshold=1.0, **noise_rates)
    try:
        results = [(run_sequence(env, seq)[0], seq) for seq in circuits]
    finally:
        env.close()
    return sorted(results, key=lambda r: r[0], reverse=True)


def meta_noise_monte_carlo(target_index, seq, threshold, gate_time, trials, seed):
    """Completion probability and F_end stats for one circuit under meta-noise.

    An episode counts as complete if fidelity crosses the threshold at any step
    (matching the env's termination check). Reseeds numpy so every target sees
    the same sequence of noise draws.
    """
    np.random.seed(seed)
    env = QuantumPrepEnv(multi_target=True, fixed_target_index=target_index,
                         gate_time=gate_time, completion_threshold=1.0, meta_noise=True)
    f_ends = []
    completions = 0
    try:
        for _ in range(trials):
            f_end, f_max = run_sequence(env, seq)
            f_ends.append(f_end)
            if f_max > threshold:
                completions += 1
    finally:
        env.close()
    return completions / trials, np.array(f_ends)


def fmt_circuit(seq):
    return " -> ".join(GATE_NAMES[a] for a in seq)


def run_multi_bell_analysis(args):
    print("=" * 72)
    print("MULTI-BELL NOISE CEILING ANALYSIS")
    print("=" * 72)
    print(f"\nWorst-case rates: {META_NOISE_MAX}")
    print(f"Mean meta rates:  {META_NOISE_MEAN}")
    print(f"gate_time={args.gate_time}  threshold={args.threshold}  "
          f"mc_trials={args.mc_trials}  seed={args.seed}")

    summary = []
    for idx, name in enumerate(QuantumPrepEnv.TARGET_NAMES):
        print(f"\n{'-' * 72}")
        print(f"Target {idx}: {name}")
        depth, circuits = find_minimal_circuits(idx, args.gate_time)
        print(f"  Minimal depth: {depth} gates | noiseless-optimal orderings: {len(circuits)}")

        worst = rank_orderings(idx, circuits, META_NOISE_MAX, args.gate_time)
        mean = rank_orderings(idx, circuits, META_NOISE_MEAN, args.gate_time)
        wc_best_f, wc_best_seq = worst[0]
        wc_worst_f, wc_worst_seq = worst[-1]
        mn_best_f, mn_best_seq = mean[0]
        mn_worst_f, mn_worst_seq = mean[-1]

        print(f"  Worst-case noise: best  F={wc_best_f:.4f}  {fmt_circuit(wc_best_seq)}")
        print(f"                    worst F={wc_worst_f:.4f}  {fmt_circuit(wc_worst_seq)}")
        print(f"                    ordering spread = {wc_best_f - wc_worst_f:.4f}")
        print(f"  Mean meta noise:  best  F={mn_best_f:.4f}  {fmt_circuit(mn_best_seq)}")
        print(f"                    worst F={mn_worst_f:.4f}  {fmt_circuit(mn_worst_seq)}")

        comp, f_ends = meta_noise_monte_carlo(
            idx, wc_best_seq, args.threshold, args.gate_time, args.mc_trials, args.seed)
        print(f"  Meta-noise MC ({args.mc_trials} eps, worst-case-best ordering): "
              f"completion@{args.threshold:.2f} = {comp * 100:.1f}%  "
              f"F_end mean={f_ends.mean():.4f}  min={f_ends.min():.4f}")

        margin = wc_best_f - args.threshold
        verdict = "CLEARS" if margin > 0.02 else ("MARGINAL" if margin > 0 else "BELOW")
        print(f"  Threshold {args.threshold:.2f} vs worst-case ceiling {wc_best_f:.4f}: "
              f"{verdict} (margin {margin:+.4f})")

        summary.append((name, depth, len(circuits), wc_best_f, wc_worst_f,
                        mn_best_f, comp, margin))

    print(f"\n{'=' * 72}")
    print("SUMMARY")
    print(f"{'=' * 72}")
    print(f"  {'target':<10} {'depth':>5} {'#opt':>5} {'wc best':>8} {'wc worst':>9} "
          f"{'mean best':>10} {'MC comp':>8} {'margin':>8}")
    for name, depth, n, wcb, wcw, mnb, comp, margin in summary:
        print(f"  {name:<10} {depth:>5} {n:>5} {wcb:>8.4f} {wcw:>9.4f} "
              f"{mnb:>10.4f} {comp * 100:>7.1f}% {margin:>+8.4f}")

    floor = min(s[3] for s in summary)
    print(f"\n  Worst-case ceiling floor across targets: {floor:.4f}")
    if floor - args.threshold > 0.02:
        print(f"  Threshold {args.threshold:.2f} clears every target's worst-case "
              f"ceiling with margin. Proceed.")
    else:
        fallback = max(round(floor - 0.05, 2), 0.5)
        print(f"  WARNING: threshold {args.threshold:.2f} does NOT clear the "
              f"worst-case ceiling floor with margin.")
        print(f"  Suggested fallback completion_threshold: {fallback:.2f}")


def run_single_target_analysis(args):
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
    parser.add_argument("--multi_bell", action="store_true",
                        help="Per-target ceiling analysis across all 4 Bell states")
    parser.add_argument("--threshold", type=float, default=0.80,
                        help="Completion threshold to validate (--multi_bell mode)")
    parser.add_argument("--mc_trials", type=int, default=200,
                        help="Meta-noise Monte Carlo episodes per target (--multi_bell mode)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the meta-noise Monte Carlo (--multi_bell mode)")
    args = parser.parse_args()

    if args.multi_bell:
        run_multi_bell_analysis(args)
    else:
        run_single_target_analysis(args)


if __name__ == "__main__":
    main()
