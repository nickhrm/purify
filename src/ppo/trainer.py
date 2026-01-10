import os

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy


def main(): 
    coherence_times  = [0.001, 0.003, 0.007, 0.01, 0.005, 0.1]
    lambdas = [(0.3, 0.0, 0.0),(0.0, 0.3, 0.0), (0.0, 0.0, 0.3),]

    for coherence_time in coherence_times:
        # First train for fixed lambdas
        for lam in lambdas:
            constants = ConstantsTuple(
                coherence_time=coherence_time,
                lambda_strategy=LambdaSrategy.USE_CONSTANTS,
                lambdas=lam,
                pumping_probability=1,
                waiting_time_sensitivity=1,
            )
            # WICHTIG: Eindeutigen Namen für Logs generieren, damit sie sich nicht überschreiben
            run_name = f"fixed_{lam}_{coherence_time}"
            train(constants, run_name)

        # Train with Random Lambdas
        constants = ConstantsTuple(
                coherence_time=coherence_time,
                lambda_strategy=LambdaSrategy.RANDOM,
                lambdas=(0,0,0),
                pumping_probability=1,
                waiting_time_sensitivity=1,
            )
        run_name = f"random_{coherence_time}"
        train(constants, run_name)


def train(constants: ConstantsTuple, run_name: str):
    num_cpu = 5 
    log_dir = "./ppo_results/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. Trainings-Umgebung (Vectorized für Speed)
    env = make_vec_env(
        lambda: TrainingEnv(constants), n_envs=num_cpu, vec_env_cls=SubprocVecEnv
    )

    eval_env = Monitor(TrainingEnv(constants))

    model_path = f"results/agent_{run_name}.zip"

    # --- CALLBACK SETUP ---
    
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10,
        verbose=1
    )

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=300000,
        callback_after_eval=stop_train_callback,
        best_model_save_path=f"./best_models/{run_name}/",
        verbose=1,
        deterministic=True
    )


    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
    )

    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"Starte neues Training für {run_name}...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            learning_rate=0.0001,
            gamma=1,
            gae_lambda=0.95,
            ent_coef=0.01,
            device="cpu",
            verbose=1,
            tensorboard_log=log_dir,
        )

    print("Starte Training mit Early Stopping...")

    try:
        model.learn(total_timesteps=50_000_000, reset_num_timesteps=False, callback=eval_callback)
    except KeyboardInterrupt:
        print("Training manuell unterbrochen...")

    model.save(model_path)
    print("Training beendet und Modell gespeichert.")


if __name__ == "__main__":
    main()