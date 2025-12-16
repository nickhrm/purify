from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env

from ppo.custom_env import TrainingEnv
# Importiere oder definiere deine TrainingEnv-Klasse

# -----------------------------------------------
# SCHRITT 1: Registrieren Sie die benutzerdefinierte Umgebung
# -----------------------------------------------

# Zuerst Ray initialisieren
ray.init(ignore_reinit_error=True)

def env_creator():
    return TrainingEnv()

# Umgebung registrieren
ENV_NAME = "QuantumEntanglementEnv"
register_env(ENV_NAME, env_creator)

# -----------------------------------------------
# SCHRITT 2: PPO konfigurieren
# -----------------------------------------------

# Annahme: Deine Action-Klasse (Action) hat N mögliche diskrete Werte (z.B. 7 Strategien).
NUM_DISCRETE_ACTIONS = 7


config = PPOConfig()
config.environment(ENV_NAME)
config.env_runners(num_env_runners=6)
config.training(
    gamma=1, lr=0.01, kl_coeff=0.3, train_batch_size_per_learner=256
)


# 3. PPO-Algorithmus bauen und trainieren
algo = config.build()

# Trainiere für eine bestimmte Anzahl von Iterationen
print("Starte das Training...")

for i in range(10):  
    result = algo.train()
    print(f"Iteration {i}: Mean Reward = {result['episode_reward_mean']:.4f}")

# Optional: Speichere den finalen Trainer
# algo.save("/pfad/zum/checkpoint")

print("Training abgeschlossen.")