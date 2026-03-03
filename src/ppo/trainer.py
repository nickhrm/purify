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
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    # Für kurze Kohärenzzeiten helfen kleinere n_steps, da Rewards seltener sind
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    vf_coef = trial.suggest_float("vf_coef", 0.3, 1.0)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    # Netzwerk-Architektur auswählen
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large", "deep_medium"])
    if net_arch_type == "small":
        net_arch = dict(pi=[64, 64], vf=[64, 64])
    elif net_arch_type == "medium":
        net_arch = dict(pi=[128, 128], vf=[128, 128])
    elif net_arch_type == "large":
        net_arch = dict(pi=[256, 256], vf=[256, 256])
    else:  # deep_medium: 3 Schichten helfen bei dieser Zustandsstruktur
        net_arch = dict(pi=[128, 128, 128], vf=[128, 128, 128])

    return {
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "batch_size": batch_size,
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "clip_range": clip_range,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
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
        gae_lambda=1,
        gamma=1,
        verbose=0 # Verbose auf 0, um die Konsole beim Hyperparameter-Tuning nicht zu fluten
    )

    # 4. Modell trainieren
    # Tipp: Für die Optimierung oft eine reduzierte Schrittzahl verwenden (z.B. 2_000_000 statt 50_000_000), 
    # damit Optuna in vernünftiger Zeit Ergebnisse liefert.
    # 1M Steps für stabilere Evaluation bei kleinen Kohärenzzeiten
    total_timesteps = 1_000_000
    
    try:
        model.learn(total_timesteps=total_timesteps)
    except Exception as e:
        print(f"Fehler während des Trainings im Trial {trial.number}: {e}")
        return 0 # Schlechter Wert bei Absturz
    finally:
        env.close()

    # 5. Evaluierung des Modells (Wie im Artikel: Test über mehrere Episoden)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=50, deterministic=True)
    eval_env.close()

    print(f"Trial {trial.number} beendet mit Mean Reward: {mean_reward} +/- {std_reward}")

    # Optuna maximiert diesen zurückgegebenen Wert
    return mean_reward


def train(constants: ConstantsTuple, num_cpu: int):

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
        max_no_improvement_evals=400,
        min_evals=200,
        verbose=1,
        param_label=constants.subfolder_name()
    )
    desired_total_steps_per_eval = 20000
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

    # ─── Analytisch hergeleitete Hyperparameter für coherence_time=0.08 ──────────
    #
    # Umgebungsstatistiken (3000 Episoden, zufällige Policy):
    #   mean_steps=660, std=475, P95=1566  →  alle Episoden terminieren
    #   Var(reward)=0.00116  (sehr gering, keine Truncation)
    #   T2/DELTA_T = 800 Steps  (Kohärenz sehr lang relativ zu Episodenlänge)
    #
    # n_steps = 4096:
    #   Mit gae_lambda=1 brauchen wir vollständige Episoden (MC-Return).
    #   4096 / 660 ≈ 6 Episoden/Rollout × 6 CPUs → ~36 Episoden pro Update.
    #   Genug für stabile Return-Schätzung ohne Bootstrapping-Fehler.
    #
    # batch_size = 512:
    #   Buffer = n_steps × num_envs = 4096 × 6 = 24576
    #   batch_size = 512 → 48 Mini-Batches/Epoche → stabiler, sauberer Gradient.
    #   Var(r) klein → große Batches OK, kein Extra-Rauschen nötig.
    #
    # n_epochs = 10:
    #   Niedrige Reward-Varianz → Daten werden nicht schnell "veraltet".
    #   10 Epochen nutzen jeden Rollout maximal aus ohne Policy-Drift.
    #
    # learning_rate = linear 3e-4 → 0:
    #   Startwert 3e-4 bewährt (vorheriger Lauf: 0.000295 war optimal).
    #   Linearer Decay verhindert Oszillation am Ende des Trainings.
    #
    # ent_coef = 0.005:
    #   4 Aktionen → max H = ln(4)≈1.39. Kleine Entropie reicht für Exploration.
    #   Nicht zu groß, da der Reward gut informativ ist (keine flachen Plateaus).
    #
    # net_arch = [256, 256]:
    #   6D Input mit nichtlinearen Purifikations-Entscheidungsgrenzen.
    #   "large" Architektur hat sich im Vorgänger-Lauf als beste erwiesen.
    # ─────────────────────────────────────────────────────────────────────────────
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
            n_steps=4096,       # ~6 Episoden/Rollout × 6 CPUs → stabile MC-Returns
            batch_size=512,     # 24576 Buffer / 512 → 48 Mini-Batches/Epoche
            n_epochs=10,        # Niedrige Reward-Varianz → viele Epochen sicher
            learning_rate=lambda progress: 3e-4 * progress,  # Linearer Decay: 3e-4 → 0
            gamma=1.0,          # Kein Discount (einmaliger Endreward)
            gae_lambda=1.0,     # MC-Return (kein Bootstrapping-Bias)
            ent_coef=0.005,     # Leichte Exploration bei 4 Aktionen
            clip_range=0.2,     # PPO-Standard, bei kleiner Reward-Varianz ausreichend
            vf_coef=0.5,        # PPO-Standard
            max_grad_norm=0.5,  # Verhindert exploding gradients
            device="cpu",
            verbose=1,
            tensorboard_log=log_dir,
        )

    print("Starte Training...")
    try:
        model.learn(
            total_timesteps=30_000_000,
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
                n_trials=50,  # Mehr Trials für bessere Abdeckung des Suchraums
                n_jobs=10      # Paralleles Testen von Trials (hier 1, da du in PPO schon Multiprocessing nutzt)
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
