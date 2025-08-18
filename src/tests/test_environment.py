import sys
import os
import gymnasium
from src.environment import QuantumPrepEnv
import numpy as np
import json
def main():
    """
    Tests the QuantumPrepEnv to ensure it's working correctly.
    
    This script will:
    1. Create an instance of the environment.
    2. Reset it to get the initial state.
    3. Take a few random actions and print the results.
    """
    print("--- Testing Quantum State Preparation Environment ---")
    
    # Create the environment with a non-zero noise level
    env = QuantumPrepEnv(render_mode='human')
    
    # Print the gate map for clarity
    print("\n--- Action Space Gate Map ---")
    print(json.dumps(env._gate_name_map, indent=2))
    print("-----------------------------\n")
    # Reset the environment and get the initial observation
    obs, info = env.reset()
    print(f"Initial Observation Shape: {obs.shape}")
    print(f"Initial Info: {info}")
    print("Initial State Vector:")
    print(env.current_state)
    env.render()
    
    # --- Take Specific Steps to Create Bell State ---
    print("\n--- Running Deterministic Test in Noisy Environment ---")
    actions_to_test = [
        (0, "Hadamard Q0"),
        (6, "CNOT (0->1)")
    ]
    
    for i, (action, action_name) in enumerate(actions_to_test):
        print(f"\n--- Step {i+1}/{len(actions_to_test)}, Action: {action} ({action_name}) ---")
       
        # Apply the action
        obs, reward, terminated, truncated, info = env.step(action)
       
        # Print the results
        print(f"Observation Shape: {obs.shape}")
        print(f"Reward: {reward:.4f}")
        print(f"Info: {info}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        print("Current State Vector:")
        print(env.current_state)
        env.render()
       
        if terminated or truncated:
            print("\nEpisode finished.")
            break
    # Add to main():
    paths_to_test = [
        # Optimal: H0 + CNOT0->1 (your current)
        {"name": "Optimal Bell", "actions": [0, 6], "expected": {"fidelity": (0.99, 1.01), "ent": (0.99, 1.01)}},
       
        # Suboptimal: Extra gates (H0 + X1 + CNOT0->1) – should reach ~1 but longer, test penalty
        {"name": "Suboptimal with Flip", "actions": [0, 3, 6], "expected": {"fidelity": (0, 1.01), "steps": 3}},
       
        # Cycle: X0 + X0 (back to start) – test delta~0, penalty for waste
        {"name": "Cycle Waste", "actions": [2, 2], "expected": {"fidelity": (0.7, 0.71), "reward_delta": "~0"}},
       
        # Phase Test: H0 + Z0 + CNOT – should make Bell variant (phase diff), fidelity~1 if target allows phases
        {"name": "Phase Variant", "actions": [0, 4, 6], "expected": {"fidelity": (0.0, 1.1)}},
       
        # Random-Like (Agent Early): Identity x3 – test penalty accumulation
        {"name": "No-Op Stall", "actions": [8, 8, 8], "expected": {"steps": 3, "reward": "<0"}},
       
        # Failure: Only CNOT without superpos – no change, low fidelity
        {"name": "No Superpos CNOT", "actions": [6], "expected": {"fidelity": (0.5, 0.8)}},

        # Probe and Prepare: Identity, Hadamard, Identity, CNOT
        {"name": "Probe and Prepare", "actions": [8, 0, 8, 6], "expected": {"fidelity": (0.99, 1.01), "steps": 4}},
    ]
    for path in paths_to_test:
        print(f"\n--- Testing Path: {path['name']} ---")
        env.reset()
        cumulative_reward = 0
        for action in path['actions']:
            obs, reward, term, trunc, info = env.step(action)
            cumulative_reward += reward
            print(f"Action {action}: Reward {reward:.4f}, Fidelity {info['fidelity']:.4f}, Ent {info['entanglement']:.4f}, Superpos {info['superpos']:.4f}")
        print(f"Path OK: Cumulative Reward {cumulative_reward:.4f}")
    print("\n--- Environment Test Complete ---")
    env.close()
if __name__ == "__main__":
    main()