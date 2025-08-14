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

    print("\n--- Environment Test Complete ---")
    env.close()

if __name__ == "__main__":
    main()
