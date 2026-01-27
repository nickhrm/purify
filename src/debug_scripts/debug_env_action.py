
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple, tupleAdapter
from purify.my_enums import LambdaSrategy, Action
from purify.my_constants import AVAILABLE_ACTIONS

def debug_env():
    print("Available Actions:", AVAILABLE_ACTIONS)
    
    constants = ConstantsTuple(
        coherence_time=0.01,
        lambda_strategy=LambdaSrategy.USE_CONSTANTS,
        lambdas=(0.3, 0.0, 0.0),
        pumping_probability=1,
        waiting_time_sensitivity=1,
    )
    
    env_config = tupleAdapter(constants)
    print("Env Config Keys:", env_config.keys())
    
    env = TrainingEnv(env_config)
    obs, _ = env.reset()
    print("Initial Obs:", obs)
    
    # Try Action 0 (REPLACE) explicitly
    print("\nAttempting Action 0 (REPLACE)...")
    try:
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"Action 0 Result: Reward={reward}, Terminated={terminated}, Info={info}")
    except Exception as e:
        print(f"Action 0 FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Try random steps
    print("\nRunning random steps...")
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample()
        print(f"Step {i}: Action {action} ({AVAILABLE_ACTIONS[action]})")
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Episode ended")
            env.reset()

if __name__ == "__main__":
    debug_env()
