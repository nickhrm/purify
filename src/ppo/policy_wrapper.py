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
            # Nutze Algorithm.from_checkpoint (funktioniert auch für Legacy Checkpoints)
            self.model = Algorithm.from_checkpoint(path)
            print("loaded Rllib Model (Legacy API)")
        except Exception as e:
            print(f"Kritischer Fehler beim Laden von Ray Checkpoint: {e}")
            self.model = None

    def predict(self, obs):
        if self.model is None:
            print("model is none")
            raise ValueError('Model is None')

        # Legacy API: compute_single_action
        # explore=False sorgt für deterministisches Verhalten (argmax über Logits)
        action = self.model.compute_single_action(obs, explore=False)
        
        # Falls compute_single_action ein Tuple zurückgibt (action, state, info),
        # müssen wir manchmal entpacken, aber meistens ist es direkt die Action,
        # wenn full_fetch=False (default).
        
        # print(action)
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
