import multiprocessing
import os

import torch
from stable_baselines3 import PPO

# VecNormalize wurde aus dem Import entfernt
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy


def make_env():
    # Deine Konstanten
    current_constants = ConstantsTuple(
        decoherence_time=0.05,
        strategy=Action.TRAINING_MODE,
        lambda_strategy=LambdaSrategy.RANDOM,
        waiting_time_sensitivity=1,
        pumping_probability=1.0,
    )
    return TrainingEnv(current_constants)


def continue_training():
    num_cpu = max(1, multiprocessing.cpu_count() - 2)

    # 1. Environment erstellen
    # Wir nennen es direkt 'env', da wir keinen Zwischenschritt mehr brauchen
    env = SubprocVecEnv([make_env for _ in range(num_cpu)])

    log_dir = "./ppo_results/"
    os.makedirs(log_dir, exist_ok=True)

    # Monitor loggt Rewards für Tensorboard (wichtig, auch ohne Normalisierung)
    env = VecMonitor(env, log_dir)

    # --- HIER WURDE VECNORMALIZE ENTFERNT ---

    model_path = "ppo_quantum_agent.zip"

    # Architektur beibehalten
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
    )

    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")

        # Modell laden ohne speziellen Wrapper
        model = PPO.load(
            "ppo_quantum_agent",
            env=env,
            device="cpu",
            # Falls du Hyperparameter im laufenden Betrieb ändern willst:
            custom_objects={"learning_rate": 0.0001, "gamma": 1, "n_steps": 2048},
        )
    else:
        print("Kein Modell gefunden. Starte neues Training...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=1,
            gae_lambda=0.95,
            ent_coef=0.01,
            device="cpu",
            verbose=1,
            tensorboard_log=log_dir,
        )

    print("Starte Training...")

    try:
        model.learn(total_timesteps=5_000_000, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training unterbrochen...")

    # 3. Speichern - Nur noch das Modell, keine Stats mehr
    model.save("ppo_quantum_agent")
    print("Modell gespeichert.")


if __name__ == "__main__":
    continue_training()