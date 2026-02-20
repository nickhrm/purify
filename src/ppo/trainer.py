import os

from ppo.custom_stop_callback import CustomStopCallback

# --- SCHRITT 1: Threading begrenzen (Muss GANZ oben stehen) ---
# Verhindert, dass Numpy/Torch pro Prozess alle 20 Cores blockieren.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from stable_baselines3 import PPO, DQN
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
    coherence_times = [0.08]

    NUM_CORES_PER_RUN = 6
    for coherence_time in coherence_times:
        constants = ConstantsTuple(
            coherence_time=coherence_time,
            lambda_strategy=LambdaSrategy.RANDOM,
            lambdas=(0.0, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
            actions=(Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3,)
        )
        print(f"Training für coherence time: {coherence_time} mit {NUM_CORES_PER_RUN} Cores")

        # train(constants, NUM_CORES_PER_RUN)
        train_dqn(constants)


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
        max_no_improvement_evals=38,
        min_evals=60,
        verbose=1,
        param_label=constants.subfolder_name()
    )
    desired_total_steps_per_eval = 50000
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
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
        activation_fn=torch.nn.Tanh,
    )


    

    if os.path.exists(model_path):
        print(f"Lade existierendes Modell: {model_path}")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print(f"Starte neues Training für {constants.folder_name()}, {constants.subfolder_name()}...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=4096,
            batch_size=2048,           # Kann evtl. auf 256 erhöht werden bei größerem Puffer
            n_epochs=10,
            learning_rate=0.0001,
            gamma=0.999,
            gae_lambda=1,
            ent_coef=0.05,
            clip_range_vf=None,
            normalize_advantage=False,
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


def train_dqn(constants: ConstantsTuple):
    # Pfade anpassen
    model_path = f"results/dqn_all/{constants.folder_name()}/{constants.subfolder_name()}"
    log_dir = f"logs/dqn/{constants.folder_name()}"
    best_model_dir = f"results/dqn_best/{constants.folder_name()}/{constants.subfolder_name()}/"
    os.makedirs(log_dir, exist_ok=True)

    # Trainings-Environment (n_envs=1 ist für DQN oft am stabilsten)
    env = make_vec_env(lambda: TrainingEnv(constants), n_envs=1)

    # --- NEU: Evaluations-Environment ---
    # Wir brauchen ein separates Env, damit die Evaluation nicht das Training stört
    eval_env = make_vec_env(lambda: TrainingEnv(constants), n_envs=1)

    # --- NEU: Callbacks einrichten ---
    stop_train_callback = CustomStopCallback(
        max_no_improvement_evals=38,
        min_evals=60,
        verbose=1,
        param_label=constants.subfolder_name()
    )
    
    # Da wir n_envs=1 haben, entsprechen die Steps direkt den total_timesteps.
    # Wir evaluieren alle 50.000 Schritte.
    eval_freq = 50000 

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=50,
        callback_after_eval=stop_train_callback,
        best_model_save_path=best_model_dir,
        verbose=1,
        deterministic=True, # Wichtig: Bei Eval wird die beste Aktion ohne Zufall gewählt
    )


    policy_kwargs = dict(
        net_arch=[128, 128],
        activation_fn=torch.nn.Tanh
    )

    # Prüfen, ob schon ein Modell existiert (optional, wie bei PPO)
    if os.path.exists(model_path + ".zip"):
        print(f"Lade existierendes DQN Modell: {model_path}")
        model = DQN.load(model_path, env=env, device="cpu")
    else:
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,
            buffer_size=100_000,        # Genug für ~100 Episoden
            learning_starts=5000,       # Genug sammeln, um sicher Rewards im Buffer zu haben
            batch_size=256,             # Etwas größer
            tau=1.0,
            target_update_interval=1000, # Etwas seltener updaten, stabilisiert das Ziel
            train_freq=(10, "step"),    # Alle 10 Schritte...
            gradient_steps=10,          # ...10 Updates (viel intensiveres Lernen vom Buffer)
            gamma=0.999,                # NICHT 1.0!
            exploration_fraction=0.3,   # Mehr Zeit zum Herumprobieren geben
            exploration_final_eps=0.05,
            device="cpu",
            verbose=1,
            tensorboard_log=log_dir,
        )
    print(f"Starte DQN Training für: {constants.subfolder_name()}...")
    
    try:
        model.learn(
            total_timesteps=10_000_000,
            callback=eval_callback, # --- NEU: Callback hier übergeben ---
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Training manuell unterbrochen...")
    finally:
        env.close()
        eval_env.close() # --- NEU: Eval Env sauber schließen ---

    model.save(model_path)
    print("DQN Modell gespeichert.")

if __name__ == "__main__":
    main()
