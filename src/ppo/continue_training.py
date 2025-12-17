import os
import multiprocessing
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from ppo.custom_env import TrainingEnv

def make_env():
    # Hier sicherstellen, dass die neue decoherence_time 
    # entweder in den Constants oder im TrainingEnv gesetzt ist
    return TrainingEnv()

def continue_training():
    # 1. Konfiguration für Parallelisierung
    num_cpu = max(1, multiprocessing.cpu_count() - 1) 
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])
    
    log_dir = "./ppo_results_continued/"
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, log_dir)

    model_path = "ppo_quantum_agent.zip"

    # 2. Modell laden oder neu erstellen
    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")
        # Wir laden das Modell und verknüpfen es mit dem neuen Env
        model = PPO.load(
            "ppo_quantum_agent", 
            env=env, 
            device='cpu',
            # Wir überschreiben die Learning Rate, um feiner nachzujustieren
            custom_objects={"learning_rate": 0.0001} 
        )
        # Optional: Tensorboard Log-Pfad aktualisieren
        model.tensorboard_log = log_dir
    else:
        print("Kein Modell gefunden. Starte neues Training...")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=1024,      
            batch_size=64,     
            n_epochs=4,        
            learning_rate=0.0003,
            gamma=1.0,
            ent_coef=0.01,
            device='cpu',
            verbose=1,
            tensorboard_log=log_dir
        )

    # 3. Weitertrainieren
    # Da die Umgebung jetzt schwieriger ist (neue Dekohärenz), 
    # braucht er vielleicht wieder ein paar Millionen Steps
    total_steps_to_add = 2_000_000
    print(f"Trainiere weiter für {total_steps_to_add} Schritte...")
    
    model.learn(
        total_timesteps=total_steps_to_add,
        reset_num_timesteps=False # WICHTIG: Damit die Timesteps im Log weiterlaufen
    )

    # 4. Speichern unter neuem Namen (oder überschreiben)
    model.save("ppo_quantum_agent_refined")
    print("Training beendet und verfeinertes Modell gespeichert.")

if __name__ == "__main__":
    continue_training()