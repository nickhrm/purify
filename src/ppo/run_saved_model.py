from math import inf
import os
import numpy as np
from stable_baselines3 import PPO
from ppo.custom_env import TrainingEnv 
from purify.my_enums import Action 

def run_saved_model():
    # 1. Pfad zum gespeicherten Modell definieren
    # SB3 fügt automatisch .zip hinzu, wenn es fehlt.
    model_path = "ppo_quantum_agent" 

    if not os.path.exists(f"{model_path}.zip"):
        print(f"Fehler: Modelldatei '{model_path}.zip' nicht gefunden!")
        return

    # 2. Environment neu erstellen
    # WICHTIG: Es muss exakt die gleiche Umgebung sein wie beim Training
    env = TrainingEnv()

    # 3. Modell laden
    print(f"Lade Modell von: {model_path}...")
    model = PPO.load(model_path, env=env)

    # 4. Loop laufen lassen
    episodes = 1000 # Wie viele Durchläufe willst du sehen?

    for ep in range(episodes):
        print(f"\n--- Starte Episode {ep + 1} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        step_counter = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Reward is {reward}")


            total_reward += reward
            step_counter += 1

            # Optional: Action lesbar machen für den Print
            # (Hängt davon ab, wie du deine Action-Liste definiert hast, z.B. so:)
            try:
                action_name = list(Action)[int(action)].name
            except:
                action_name = str(action)

            # Nur interessante Schritte printen (z.B. wenn Reward > 0 oder alle 10 Schritte)
            if reward > 0 or step_counter % 1 == 0: # Hier % 1 für ALLES printen
                print(f"Step {step_counter:3} | Action: {action} ({action_name}) | Reward: {reward:.4f} | Obs: {obs}")
                print(f"2nd event: {info["event"]}, reward: {info["reward"]}")

            done = terminated or truncated

        print(f"Episode {ep + 1} beendet. Total Reward: {total_reward:.4f}")

if __name__ == "__main__":
    run_saved_model()