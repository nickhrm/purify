import os
import json
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.custom_env import TrainingEnv
from ppo.custom_stop_callback import CustomStopCallback
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy



def save_training_info(path: str, constants: ConstantsTuple, hyperparams: dict, extra: dict | None = None):
    """Speichert eine JSON-Datei mit allen Trainingsinfos neben das Modell."""
    os.makedirs(path, exist_ok=True)
    info = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "constants": {
            "coherence_time": constants.coherence_time,
            "pumping_probability": constants.pumping_probability,
            "waiting_time_sensitivity": constants.waiting_time_sensitivity,
            "lambda_strategy": constants.lambda_strategy.name,
            "lambdas": list(constants.lambdas),
            "actions": [a.name for a in constants.actions],
        },
        "hyperparams": hyperparams,
    }
    if extra:
        info.update(extra)
    info_path = os.path.join(path, "training_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"📋 Trainingsinfo gespeichert: {info_path}")


def load_training_info(path: str) -> dict | None:
    """Liest die training_info.json und gibt sie als Dict zurück (oder None)."""
    info_path = os.path.join(path, "training_info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_training_info(info: dict):
    """Gibt die gespeicherten Trainingsinfos leserlich aus."""
    print("\n" + "─" * 55)
    print(f"  📅 Trainiert am : {info.get('trained_at', 'unbekannt')}")
    c = info.get("constants", {})
    print(f"  🔬 Kohärenzzeit  : {c.get('coherence_time')}")
    print(f"  🎲 λ-Strategie   : {c.get('lambda_strategy')}  λ={c.get('lambdas')}")
    print(f"  ⚙️  Aktionen       : {c.get('actions')}")
    hp = info.get("hyperparams", {})
    print("  📐 Hyperparameter:")
    for k, v in hp.items():
        if k != "policy_kwargs":
            print(f"      {k}: {v}")
    if "policy_kwargs" in hp:
        print(f"      policy_kwargs: {hp['policy_kwargs']}")
    if "total_timesteps" in info:
        print(f"  🏃 Schritte      : {info['total_timesteps']:,}")
    if "mean_reward" in info:
        print(f"  🏆 Mean Reward   : {info['mean_reward']:.4f} ± {info.get('std_reward', '?')}")
    print("─" * 55 + "\n")


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

    # ─── Optuna-gefundene Hyperparameter (bestes Trial) ──────────────────────────
    #
    # Quelle: Optuna-Studie, beste Parameter:
    #   learning_rate ≈ 2.95e-4, ent_coef ≈ 0.003, gae_lambda=0.99,
    #   batch_size=256, n_steps=1024, n_epochs=6, net_arch='large'
    #
    # Anpassung für gae_lambda=1 (reines MC-Return, kein Bootstrapping):
    #   n_steps: 1024 → 2048  – mehr Puffer für vollständige Episoden
    #   (mean_steps≈660 → ~3 Episoden/Rollout/Env × 6 Envs = ~18 Episoden
    #    pro Update; ausreichend für stabile MC-Return-Schätzung)
    #   Alle anderen Parameter bleiben identisch zum Optuna-Ergebnis.
    # ─────────────────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # net_arch='large'
        activation_fn=torch.nn.Tanh,
    )


    # Hyperparameter als typisierte Variablen – wird auch ins JSON gespeichert
    hp_n_steps: int          = 1024   # Optuna: 1024; ≥1 Episoden sicher mit gae_lambda=1
    hp_batch_size: int       = 256    # Optuna-Wert
    hp_n_epochs: int         = 6      # Optuna-Wert
    hp_learning_rate: float  = 2.9496e-4  # Optuna-Wert (fest, kein Decay)
    hp_gamma: float          = 1.0
    hp_gae_lambda: float     = 0.99    # MC-Return (Optuna: 0.99, User will 1.0)
    hp_ent_coef: float       = 0.003011806764086755  # Optuna-Wert
    hp_clip_range: float     = 0.2
    hp_vf_coef: float        = 0.5
    hp_max_grad_norm: float  = 0.5

    # Dict für die JSON-Ausgabe
    hyperparams = dict(
        n_steps=hp_n_steps,
        batch_size=hp_batch_size,
        n_epochs=hp_n_epochs,
        learning_rate=hp_learning_rate,
        gamma=hp_gamma,
        gae_lambda=hp_gae_lambda,
        ent_coef=hp_ent_coef,
        clip_range=hp_clip_range,
        vf_coef=hp_vf_coef,
        max_grad_norm=hp_max_grad_norm,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn="Tanh"),
    )

    if os.path.exists(best_model_dir):
        print(f"Lade existierendes Modell: {best_model_dir}")
        model = PPO.load(best_model_dir + "/best_model.zip", env=env, device="cpu")
        info = load_training_info(best_model_dir)
        if info:
            print("📂 Vorhandenes Modell – gespeicherte Trainingsinfos:")
            print_training_info(info)
        else:
            print("⚠️  Keine training_info.json gefunden (altes Modell).")
    else:
        print(f"Starte neues Training für {constants.folder_name()}, {constants.subfolder_name()}...")
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            n_steps=hp_n_steps,
            batch_size=hp_batch_size,
            n_epochs=hp_n_epochs,
            learning_rate=hp_learning_rate,
            gamma=hp_gamma,
            gae_lambda=hp_gae_lambda,
            ent_coef=hp_ent_coef,
            clip_range=hp_clip_range,
            vf_coef=hp_vf_coef,
            max_grad_norm=hp_max_grad_norm,
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

    # Trainingsinfos neben das finale Modell speichern
    save_training_info(
        path=os.path.dirname(model_path),
        constants=constants,
        hyperparams=hyperparams,
        extra={"total_timesteps": 30_000_000},
    )
    # ... und neben das beste Modell (wird vom EvalCallback dort abgelegt)
    save_training_info(
        path=best_model_dir,
        constants=constants,
        hyperparams=hyperparams,
        extra={"total_timesteps": 30_000_000},
    )



def main():
    coherence_times = [0.01, 0.05, 0.09]
    NUM_CORES_PER_RUN = 6

    for coherence_time in coherence_times:
        constants = ConstantsTuple(
            coherence_time=coherence_time,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
            actions=(Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3,Action.PMD)
        )
        print(f"Lauf für coherence time: {coherence_time} mit {NUM_CORES_PER_RUN} Cores")
        train(constants, NUM_CORES_PER_RUN)


if __name__ == "__main__":
    main()
