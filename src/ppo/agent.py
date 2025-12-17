import os
import multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from ppo.custom_env import TrainingEnv

def make_env():
    return TrainingEnv()

def train_agent():
    # 1. Maximale CPU-Auslastung
    # Nutze fast alle Kerne, lass 1-2 für das System frei, um Ruckeln zu vermeiden
    num_cpu = max(1, multiprocessing.cpu_count() - 1) 
    
    # SubprocVecEnv ist perfekt für CPU-Parallelisierung
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    log_dir = "./ppo_results/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    # 2. CPU-Optimierte Hyperparameter
    model = PPO(
        "MlpPolicy",
        env,
        # Kleine n_steps für CPU: Agent lernt öfter aus frischen Daten
        n_steps=1024,      
        # Kleinere Batch-Size: Die CPU verarbeitet kleine Häppchen schneller als riesige Matrizen
        batch_size=64,     
        # Weniger Epochen: CPU braucht länger für Optimierungsschritte, 
        # daher reduzieren wir die Wiederholungen pro Batch
        n_epochs=4,        
        learning_rate=0.0003,
        gamma=1.0,
        ent_coef=0.01,
        device='cpu',      # Explizit CPU nutzen
        verbose=1,
        tensorboard_log=log_dir
    )

    print(f"Starte CPU-optimiertes Training mit {num_cpu} Workern...")
    model.learn(total_timesteps=5_000_000)
    model.save("ppo_quantum_agent")

if __name__ == "__main__":
    train_agent()