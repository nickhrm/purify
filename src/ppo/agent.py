import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from ppo.custom_env import TrainingEnv

# Importieren Sie Ihre oben definierte Klasse
# from my_env_file import TrainingEnv 

def train_agent():
    # 1. Environment erstellen
    env = TrainingEnv()
    
    # 2. Sanity Check (EXTREM NÜTZLICH)
    # SB3 prüft hier, ob Ihr Env den Gym-Standards entspricht (Shapes, Datentypen, Limits).
    # Wenn hier ein Fehler kommt, stimmt etwas in der Env-Klasse nicht.
    print("Prüfe Environment...")
    check_env(env) 
    print("Environment ist valide!")

    # 3. Verzeichnis für Logs erstellen
    log_dir = "./ppo_results/"
    os.makedirs(log_dir, exist_ok=True)

    # 4. Agent initialisieren
    # MlpPolicy = Standard neuronales Netz für Vektor-Input
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        gamma=1.0,  # Ihr gewünschtes Gamma
        ent_coef=0.01, # Ihr gewünschter Entropy Coeff
        device='cpu'
    )

    # 5. Training starten
    print("Starte Training...")
    # total_timesteps: Wie viele Schritte (Steps) insgesamt interagiert werden soll
    model.learn(total_timesteps=300000) 

    # 6. Modell speichern
    model.save("ppo_quantum_agent")
    print("Training beendet und Modell gespeichert.")

    # --- Optional: Testen des trainierten Agenten ---
    print("Teste den trainierten Agenten...")
    obs, _ = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Obs: {obs}")
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    train_agent()