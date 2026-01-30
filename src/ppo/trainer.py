import os

from ppo.custom_stop_callback import CustomStopCallback

# --- SCHRITT 1: Threading begrenzen (Muss GANZ oben stehen) ---
# Verhindert, dass Numpy/Torch pro Prozess alle 20 Cores blockieren.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy


# 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 
def main():
    coherence_times = [0.07, 0.08, 0.09, 0.1]


    NUM_CORES_PER_RUN = 18

    for coherence_time in coherence_times:
        constants = ConstantsTuple(
            coherence_time=coherence_time,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
            actions=(Action.REPLACE, Action.PMD)
        )
        print(f"Training für coherence time: {coherence_time} mit {NUM_CORES_PER_RUN} Cores")

        train(constants, num_cpu=NUM_CORES_PER_RUN)


def train(constants: ConstantsTuple,  num_cpu: int):

    model_path = f"results/all/{constants.folder_name()}/{constants.subfolder_name()}"

    log_dir = f"logs/{constants.folder_name()}"
    os.makedirs(log_dir, exist_ok=True)

    # Erstelle die vektorisierte Umgebung
    # SubprocVecEnv ist korrekt für rechenintensive Simulationen
    env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
    )

    # Eval Env braucht nur 1 Core (oder du nutzt auch hier Subproc für Isolation)
    eval_env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=1,
        vec_env_cls=SubprocVecEnv
    )

    best_model_dir = f"results/best/{constants.folder_name()}/{constants.subfolder_name()}/"

    stop_train_callback = CustomStopCallback(
        max_no_improvement_evals=12,
        min_evals=20,
        verbose=1,
        param_label=constants.subfolder_name()
    )
    desired_total_steps_per_eval = 200000
    actual_eval_freq = max(1, desired_total_steps_per_eval // num_cpu)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=actual_eval_freq,
        n_eval_episodes=50,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_dir,
        verbose=1,
        deterministic=True,
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh,
    )

    # Ziel: Buffergröße (n_envs * n_steps) sollte ähnlich bleiben (~8192).
    # Bei 18 CPUs ist 512 eine gute Wahl (18 * 512 = 9216 Steps pro Update).
    n_steps_per_env = 512

    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"Starte neues Training für {constants.folder_name()}, {constants.subfolder_name()}...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=n_steps_per_env,  # Verringert, da n_envs erhöht wurde
            batch_size=128,           # Kann evtl. auf 256 erhöht werden bei größerem Puffer
            n_epochs=10,
            learning_rate=0.0001,
            gamma=1,
            gae_lambda=0.95,
            ent_coef=0.01,
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

if __name__ == "__main__":
    main()
