import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gymnasium as gym
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
    
    # Create the environment
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
    
    # Take a few random steps
    num_steps = 5
    for i in range(num_steps):
        # Sample a random action from the action space
        action = env.action_space.sample()
        action_name = env._gate_name_map[action]
        print(f"\n--- Step {i+1}/{num_steps}, Action: {action} ({action_name}) ---")
        
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
            print("\nEpisode finished. Resetting environment.")
            obs, info = env.reset()
            print("Initial State Vector:")
            print(env.current_state)
            env.render()

    print("\n--- Environment Test Complete ---")
    env.close()

if __name__ == "__main__":
    main()
