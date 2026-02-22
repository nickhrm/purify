import os

# --- SCHRITT 1: Threading begrenzen (Muss GANZ oben stehen) ---
# Verhindert, dass Numpy/Torch pro Prozess alle 20 Cores blockieren.
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.custom_env import TrainingEnv
from ppo.custom_stop_callback import CustomStopCallback
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy


# 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 
import os
import torch
import optuna
import numpy as np

# --- SCHRITT 1: Threading begrenzen (Muss GANZ oben stehen) ---
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Beachte, dass deine lokalen Importe hier korrekt sein müssen:
from ppo.custom_env import TrainingEnv
from ppo.custom_stop_callback import CustomStopCallback
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy

def sample_ppo_params(trial: optuna.Trial):
    """
    Diese Funktion definiert den Suchraum für Optuna.
    Hier wählt Optuna für jeden Trial neue Parameterkombinationen.
    """
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00001, 0.5, log=True)
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096, 8192])
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    
    # Netzwerk-Architektur auswählen
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    else:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    return {
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "gamma": 1,
        "gae_lambda": gae_lambda,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "policy_kwargs": dict(
            net_arch=net_arch,
            activation_fn=torch.nn.Tanh,
        ),
    }

def objective(trial: optuna.Trial, constants: ConstantsTuple, num_cpu: int):
    """
    Das Ziel für Optuna: Modell trainieren und evaluieren.
    """
    # 1. Parameter für diesen Versuch vorschlagen lassen
    kwargs = sample_ppo_params(trial)

    # 2. Environments erstellen
    env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )

    # 3. Modell mit den vorgeschlagenen Parametern erstellen
    model = PPO(
        "MlpPolicy",
        env,
        **kwargs,
        device="cpu",
        verbose=0 # Verbose auf 0, um die Konsole beim Hyperparameter-Tuning nicht zu fluten
    )

    # 4. Modell trainieren
    # Tipp: Für die Optimierung oft eine reduzierte Schrittzahl verwenden (z.B. 2_000_000 statt 50_000_000), 
    # damit Optuna in vernünftiger Zeit Ergebnisse liefert.
    total_timesteps = 500_000 
    
    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"Fehler während des Trainings im Trial {trial.number}: {e}")
        return -10000.0 # Schlechter Wert bei Absturz
    finally:
        env.close()

    # 5. Evaluierung des Modells (Wie im Artikel: Test über mehrere Episoden)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    eval_env.close()

    print(f"Trial {trial.number} beendet mit Mean Reward: {mean_reward} +/- {std_reward}")

    # Optuna maximiert diesen zurückgegebenen Wert
    return mean_reward

def main():
    coherence_times = [0.08]
    NUM_CORES_PER_RUN = 6
    
    # Willst du Optuna laufen lassen oder normal trainieren? 
    # Hier ein Switch:
    OPTIMIZE_HYPERPARAMS = False 

    for coherence_time in coherence_times:
        constants = ConstantsTuple(
            coherence_time=coherence_time,
            lambda_strategy=LambdaSrategy.RANDOM,
            lambdas=(0.0, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
            actions=(Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3,)
        )
        print(f"Lauf für coherence time: {coherence_time} mit {NUM_CORES_PER_RUN} Cores")

        if OPTIMIZE_HYPERPARAMS:
            print("Starte Optuna Hyperparameter Optimierung...")
            # Optuna Study erstellen (Ziel: Maximiere den Return/Reward)
            study = optuna.create_study(direction="maximize")
            
            # Lambda-Funktion, um unsere Argumente an die Objective-Funktion zu übergeben
            study.optimize(
                lambda trial: objective(trial, constants, NUM_CORES_PER_RUN), 
                n_trials=30,  # Anzahl der zu testenden Parameterkombinationen
                n_jobs=1      # Paralleles Testen von Trials (hier 1, da du in PPO schon Multiprocessing nutzt)
            )

            print("\n==================================")
            print("OPTIMIERUNG BEENDET")
            print("Beste gefundene Parameter:")
            print(study.best_params)
            print(f"Bester Reward: {study.best_value}")
            print("==================================\n")
            
            # TODO: Du kannst dir diese Parameter nun kopieren und fest in deine `train()`-Funktion eintragen.
        else:
            # Normales Training mit festen Parametern
            train(constants, NUM_CORES_PER_RUN)
            # train_dqn(constants)

# [... deine bestehenden Funktionen train() und train_dqn() hier ...]

if __name__ == "__main__":
    main()
def train(constants: ConstantsTuple,  num_cpu: int):

    model_path = f"results/all/{constants.folder_name()}/{constants.subfolder_name()}"

    log_dir = f"logs/{constants.folder_name()}"
    os.makedirs(log_dir, exist_ok=True)

    env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
    )

    eval_env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )

    best_model_dir = f"results/best/{constants.folder_name()}/{constants.subfolder_name()}/"

    stop_train_callback = CustomStopCallback(
        max_no_improvement_evals=160,
        min_evals=200,
        verbose=1,
        param_label=constants.subfolder_name()
    )
    desired_total_steps_per_eval = 40000
    actual_eval_freq = max(1, desired_total_steps_per_eval // num_cpu)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=actual_eval_freq,
        n_eval_episodes=70,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_dir,
        verbose=1,
        deterministic=True,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )



    if os.path.exists(best_model_dir):
        print(f"Lade existierendes Modell: {best_model_dir}")
        model = PPO.load(best_model_dir+"/best_model.zip", env=env, device="cpu")
    else:
        print(f"Starte neues Training für {constants.folder_name()}, {constants.subfolder_name()}...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=1024,
            batch_size=256,
            n_epochs=6,
            learning_rate=0.0002949643558095302,
            gamma=1,
            gae_lambda=0.99,
            ent_coef=0.003011806764086755,
            device="cpu",
            verbose=1,
            tensorboard_log=log_dir,
        )

    print("Starte Training...")
    try:
        model.learn(
            total_timesteps=50_000_000,
            reset_num_timesteps=False,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        print("Training manuell unterbrochen...")
    finally:
        # Wichtig: Envs schließen, um Prozesse zu killen
        env.close()
        eval_env.close()

    model.save(model_path)
    print("Training beendet und Modell gespeichert.")

