from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Importiere deine Env-Klasse
from ppo.custom_env import TrainingEnv
from purify.my_enums import Action


# --- Wrapper Klassen (wie zuvor besprochen) ---
class PolicyWrapper:
    def __init__(self, name): self.name = name
    def predict(self, obs) -> Any: raise NotImplementedError

class PPOAgent(PolicyWrapper):
    def __init__(self, path, env):
        super().__init__("PPO AI")
        self.model = PPO.load(path, env=env, device="cpu")
    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
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
