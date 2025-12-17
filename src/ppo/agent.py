import os
# Importiere die notwendigen Vektorisierungs-Tools
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from ppo.custom_env import TrainingEnv

def make_env():
    """Hilfsfunktion zum Erstellen einer Env-Instanz."""
    return TrainingEnv()

def train_agent():
    # 1. Konfiguration für mehrere Worker
    num_cpu = 6  # Entspricht 'num_workers' in deinem Ray-Beispiel
    
    # Erstelle das vektorisierte Environment
    # SubprocVecEnv führt jede Instanz in einem eigenen Prozess aus
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    # Monitor für vektorisierte Envs hinzufügen, um Statistiken zu loggen
    log_dir = "./ppo_results/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    # 2. Agent initialisieren
    # SB3 passt n_steps automatisch pro Worker an. 
    # Wenn n_steps=2048 und du 6 Worker hast, beträgt der Batch-Size 12288.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        gamma=1.0,
        ent_coef=0.01,
        device='auto'
    )

    # 3. Training starten
    print(f"Starte Training mit {num_cpu} Workern...")
    model.learn(total_timesteps=1000000)

    # 4. Modell speichern
    model.save("ppo_quantum_agent")
    print("Training beendet.")

if __name__ == "__main__":
    train_agent()