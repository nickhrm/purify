import os
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms import Algorithm
from stable_baselines3 import PPO

# Importiere deine Env-Klasse
from purify.my_enums import Action


# --- Wrapper Klassen (wie zuvor besprochen) ---
class PolicyWrapper:
    def __init__(self, name): self.name = name
    def predict(self, obs) -> Any: raise NotImplementedError

class SB3Agent(PolicyWrapper):
    def __init__(self, path, env):
        super().__init__("PPO AI")
        self.model = PPO.load(path, env=env, device="cpu")
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return action


class RLlibAgent(PolicyWrapper):
    def __init__(self, checkpoint_path, name="RLlib PPO"):
        super().__init__(name)
        print(f"Lade Ray Modell aus: {checkpoint_path}")
        try:
            path = os.path.join(os.getcwd(), checkpoint_path)
            self.model = Algorithm.from_checkpoint(path)
            # Pre-fetch module to ensure compatibility
            self.module = self.model.get_module()
            print("loaded Rllib Model")
        except Exception as e:
            print(f"Kritischer Fehler beim Laden von Ray Checkpoint: {e}")
            self.model = None
            self.module = None

    def predict(self, obs):
        if self.module is None:
            print("model is none")
            raise ValueError('Model is None')

        # New API Stack Inference (RLModule)
        import torch
        
        # Ensure obs is a numpy array and float32
        obs_tensor = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
        
        with torch.no_grad():
            # Run inference
            # RLModule expects a dict with "obs" key by default for PPO
            input_dict = {"obs": obs_tensor}
            output = self.module.forward_inference(input_dict)
            
            # Extract action
            # output["action_dist_inputs"] contains logits
            logits = output["action_dist_inputs"]
            # Deterministic: Argmax
            action = torch.argmax(logits, dim=1).item()
            
        print(action)
        return action


class FixedActionAgent(PolicyWrapper):
    def __init__(self, action: Action):
        super().__init__(f"Always {action.name}")
        self.action = action.value
    def predict(self, obs): return self.action

class RandomAgent(PolicyWrapper):
    def __init__(self, action_space):
        super().__init__("Random")
        self.aspace = action_space
    def predict(self, obs): return self.aspace.sample()
