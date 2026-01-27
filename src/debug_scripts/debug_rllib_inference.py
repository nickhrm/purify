import os
import torch
import numpy as np
from ray.rllib.algorithms import Algorithm

# Define path
checkpoint_path = "/home/nick/Documents/purify/ppo_results_rllib/checkpoints/0_1/best_model"

try:
    print(f"Loading Algorithm from {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    print("Getting module...")
    module = algo.get_module()
    print("Module class:", type(module))
    
    # Create dummy observation
    # Observation space is Box(0, 1, (6,), float32) based on previous files seen
    obs = np.array([0.5] * 6, dtype=np.float32)
    
    # Convert to torch tensor with batch dimension
    # RLModule expects a dict of tensors usually for 'obs'
    input_dict = {"obs": torch.from_numpy(obs).unsqueeze(0)}
    
    print("Running forward_inference...")
    with torch.no_grad():
        # Using forward_inference for action computation
        # Note: This returns a dictionary of outputs
        output = module.forward_inference(input_dict)
    
    print("Output keys:", output.keys())
    if "action_dist_inputs" in output:
        print("Action dist inputs shape:", output["action_dist_inputs"].shape)
        # For PPO/Categorical, we typically take argmax or sample
        # If deterministic, we might just take argmax of logits
        logits = output["action_dist_inputs"]
        action = torch.argmax(logits, dim=1).item()
        print("Predicted Action (Argmax):", action)
        
except Exception as e:
    print("An error occurred:", e)
    import traceback
    traceback.print_exc()
