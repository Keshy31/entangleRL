"""
Sanity-check script for QuantumPrepEnv.

Runs deterministic gate sequences and verifies that the environment
produces the expected fidelity and reward values. This validates that
the observation space, reward function, and termination logic work
correctly before any training is attempted.
"""
from src.environment.quantum_env import QuantumPrepEnv
import numpy as np


def run_path(env, name, actions, expect_fidelity_min=None, expect_terminated=None):
    obs, info = env.reset()
    cumulative_reward = 0.0
    terminated = False

    print(f"\n--- {name} ---")
    print(f"  Initial obs shape: {obs.shape}  (expect 17)")
    print(f"  Initial fidelity:  {info['fidelity']:.4f}")

    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        gate = env._gate_name_map[action]
        print(f"  Step {i+1}: {gate:15s}  fid={info['fidelity']:.4f}  "
              f"rew={reward:+.4f}  term={terminated}")

        if terminated or truncated:
            break

    print(f"  Cumulative reward: {cumulative_reward:+.4f}")
    print(f"  Final fidelity:    {info['fidelity']:.4f}")
    print(f"  Final entanglement:{info['entanglement']:.4f}")

    if expect_fidelity_min is not None:
        assert info['fidelity'] >= expect_fidelity_min, (
            f"Expected fidelity >= {expect_fidelity_min}, got {info['fidelity']:.4f}")
    if expect_terminated is not None:
        assert terminated == expect_terminated, (
            f"Expected terminated={expect_terminated}, got {terminated}")

    return info, cumulative_reward


def main():
    env = QuantumPrepEnv()

    print("=== QuantumPrepEnv Sanity Check ===")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Max steps:         {env.max_steps}")

    # 1. Optimal Bell state: H(Q0) + CNOT(0->1) -> fidelity ~1.0, should terminate
    run_path(env, "Optimal Bell (H0 + CNOT01)",
             actions=[0, 6],
             expect_fidelity_min=0.95,
             expect_terminated=True)

    # 2. No-ops only -- should NOT terminate, low fidelity
    run_path(env, "Identity x5 (no progress)",
             actions=[8, 8, 8, 8, 8],
             expect_terminated=False)

    # 3. Cancelling moves: X0 then X0 again (back to start)
    run_path(env, "X0 + X0 (cancel out)",
             actions=[2, 2],
             expect_terminated=False)

    # 4. Verify observation contains two-qubit correlators
    obs, _ = env.reset()
    assert obs.shape == (17,), f"Obs shape should be (17,), got {obs.shape}"
    # For |00>: <ZZ> should be +1, <XX>=<YY>=0
    zz_idx = 6 + 8  # ZZ is the last correlator (index 14 in obs)
    print(f"\n--- Observation structure check (|00> state) ---")
    print(f"  <Z0>={obs[2]:.3f} (expect +1)")
    print(f"  <Z1>={obs[5]:.3f} (expect +1)")
    print(f"  <ZZ>={obs[zz_idx]:.3f} (expect +1)")
    print(f"  <XX>={obs[6]:.3f} (expect 0)")
    print(f"  Fidelity in obs={obs[15]:.3f}")
    print(f"  Step/max in obs={obs[16]:.3f} (expect 0)")

    # After H0 + CNOT -> Bell state: <XX>=+1, <YY>=-1, <ZZ>=+1
    obs, _ = env.reset()
    obs, _, _, _, info = env.step(0)  # H0
    obs, _, _, _, info = env.step(6)  # CNOT01
    print(f"\n--- Observation structure check (Bell state) ---")
    print(f"  <X0>={obs[0]:.3f} (expect 0)")
    print(f"  <Z0>={obs[2]:.3f} (expect 0)")
    print(f"  <XX>={obs[6]:.3f} (expect +1)")
    xy_idx = 6 + 1  # XY
    yy_idx = 6 + 4  # YY
    print(f"  <XY>={obs[xy_idx]:.3f} (expect 0)")
    print(f"  <YY>={obs[yy_idx]:.3f} (expect -1)")
    print(f"  <ZZ>={obs[zz_idx]:.3f} (expect +1)")
    print(f"  Fidelity in obs={obs[15]:.3f} (expect ~1)")

    # 5. Verify cumulative reward is positive for optimal path
    info, cum_reward = run_path(env, "Reward check: optimal path should be highly positive",
                                 actions=[0, 6])
    assert cum_reward > 4.0, f"Expected cumulative reward > 4.0 for optimal path, got {cum_reward:.4f}"

    # 6. Verify new info fields exist
    assert "max_fidelity" in info, "Missing 'max_fidelity' in info"
    assert "completed" in info, "Missing 'completed' in info"
    assert "episode_return" in info, "Missing 'episode_return' in info"
    assert info["completed"] == 1.0, f"Expected completed=1.0 for optimal path, got {info['completed']}"
    print(f"\n--- New info fields check ---")
    print(f"  max_fidelity:  {info['max_fidelity']:.4f}")
    print(f"  completed:     {info['completed']}")
    print(f"  episode_return:{info['episode_return']:+.4f}")

    # 7. Configurable completion_threshold
    env_strict = QuantumPrepEnv(completion_threshold=0.999)
    obs, _ = env_strict.reset()
    obs, _, terminated, _, _ = env_strict.step(0)  # H0
    obs, _, terminated, _, info = env_strict.step(6)  # CNOT01 -> F~1.0
    assert terminated, "Expected termination at threshold=0.999 for perfect Bell state"
    env_strict.close()

    env_low = QuantumPrepEnv(completion_threshold=0.40)
    obs, _ = env_low.reset()
    # Initial F=0.5 > 0.40, so first action that doesn't lower fidelity should terminate
    # Actually, completion is checked after step, and initial F=0.5 > 0.40.
    # The identity gate keeps F at 0.5, which is > 0.40, so it should terminate.
    obs, _, terminated, _, _ = env_low.step(8)  # Identity -> F stays at 0.5
    assert terminated, "Expected termination at threshold=0.40 when F=0.5"
    env_low.close()
    print(f"  completion_threshold override: OK")

    env.close()

    # 8. Noisy environment: noise degrades fidelity below 1.0
    print(f"\n--- Noisy environment check ---")
    noisy_env = QuantumPrepEnv(
        amplitude_damping_rate=0.05,
        dephasing_rate=0.02,
        depolarizing_rate=0.01,
        gate_time=0.1,
        completion_threshold=1.0,  # prevent early termination
    )
    obs, info = noisy_env.reset()
    print(f"  Initial fidelity (noisy): {info['fidelity']:.4f} (expect 0.5, no noise at reset)")

    obs, _, _, _, info = noisy_env.step(0)  # H0
    fid_after_h = info['fidelity']
    print(f"  After H(Q0): fid={fid_after_h:.4f}")

    obs, _, _, _, info = noisy_env.step(6)  # CNOT01
    fid_after_cnot = info['fidelity']
    print(f"  After CNOT:  fid={fid_after_cnot:.4f}")

    assert fid_after_cnot < 1.0, (
        f"Expected fidelity < 1.0 under noise, got {fid_after_cnot:.4f}")
    assert fid_after_cnot > 0.5, (
        f"Expected fidelity > 0.5 (noise shouldn't destroy the circuit), got {fid_after_cnot:.4f}")
    print(f"  Noise degradation: {1.0 - fid_after_cnot:.4f} (fidelity loss from ideal)")

    # Verify fidelity decays further with idle steps under noise
    obs, _, _, _, info_idle = noisy_env.step(8)  # Identity (idle under noise)
    fid_idle = info_idle['fidelity']
    print(f"  After idle:  fid={fid_idle:.4f} (expect further decay)")
    assert fid_idle <= fid_after_cnot + 1e-6, (
        f"Fidelity should not increase under noise with identity gate")

    noisy_env.close()
    print(f"  Noisy environment: OK")

    print("\n=== ALL SANITY CHECKS PASSED ===")


if __name__ == "__main__":
    main()
