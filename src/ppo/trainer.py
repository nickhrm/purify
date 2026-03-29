from sqlalchemy import false
import os
import json
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.custom_stop_callback import CustomStopCallback
from purify.constants_tuple import ConstantsTuple


# ─── Pfad-Helfer ───────────────────────────────────────────────────────────────


def case_study_root(case_study_id: int) -> str:
    return f"results/case_study_{case_study_id}"


def coherence_time_folder(case_study_id: int, coherence_time: float) -> str:
    ct_str = CaseStudy.coherence_time_str(coherence_time)
    return os.path.join(case_study_root(case_study_id), ct_str)


# ─── training_info.json ────────────────────────────────────────────────────────


def save_training_info(case_study: CaseStudy) -> None:
    """
    Schreibt einmalig eine training_info.json in den Case-Study-Root-Ordner.
    Enthält die gemeinsamen Constants und Hyperparameter (NICHT die coherence_time,
    da diese je Unterordner variiert).
    Überschreibt eine existierende Datei, falls vorhanden.
    """
    root = case_study_root(case_study.id)
    os.makedirs(root, exist_ok=True)

    c = case_study.constants
    info = {
        "case_study_id": case_study.id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": "coherence_time ist nicht hier gespeichert – sie variiert je Unterordner.",
        "constants": {
            "pumping_probability": c.pumping_probability,
            "waiting_time_sensitivity": c.waiting_time_sensitivity,
            "lambda_strategy": c.lambda_strategy.name,
            "lambdas": list(c.lambdas),
            "actions": [a.name for a in c.actions],
            "min_fidelity": c.min_fidelity,
            "max_fidelity": c.max_fidelity,
        },
        "coherence_times": case_study.coherence_times,
        "hyperparams": case_study.hyperparams.to_dict(),
    }

    info_path = os.path.join(root, "training_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"📋 training_info.json gespeichert: {info_path}")


def load_training_info(case_study_id: int) -> dict | None:
    info_path = os.path.join(case_study_root(case_study_id), "training_info.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_training_info(info: dict) -> None:
    print("\n" + "─" * 55)
    print(f"  📅 Erstellt am    : {info.get('created_at', 'unbekannt')}")
    print(f"  🔢 Case Study ID  : {info.get('case_study_id')}")
    c = info.get("constants", {})
    print(f"  🎲 λ-Strategie    : {c.get('lambda_strategy')}  λ={c.get('lambdas')}")
    print(f"  📐 Fidelity-Range : [{c.get('min_fidelity')}, {c.get('max_fidelity')}]")
    print(f"  ⚙️  Aktionen        : {c.get('actions')}")
    print(f"  🕐 Kohärenzzeiten  : {info.get('coherence_times')}")
    hp = info.get("hyperparams", {})
    print("  📐 Hyperparameter:")
    for k, v in hp.items():
        if k != "policy_kwargs":
            print(f"      {k}: {v}")
    print("─" * 55 + "\n")


# ─── Training ─────────────────────────────────────────────────────────────────


def train(case_study_id: int, coherence_time: float, num_cpu: int) -> None:
    """
    Trainiert ein Modell für eine bestimmte Case Study und coherence_time.
    Jedes Gerät ruft diese Funktion für seine Teilmenge der Kohärenzzeiten auf.

    Ordnerstruktur:
      results/case_study_{id}/
        training_info.json          ← einmalig pro Case Study
        T{coherence_time_str}/
          best_model.zip            ← vom EvalCallback gespeichert
          end_model.zip             ← nach dem Training gespeichert
    """
    case_study = CASE_STUDIES[case_study_id]
    constants = case_study.make_constants(coherence_time)

    ct_folder = coherence_time_folder(case_study_id, coherence_time)
    log_dir = f"logs/case_study_{case_study_id}/{CaseStudy.coherence_time_str(coherence_time)}"

    os.makedirs(ct_folder, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # training_info.json einmalig für die gesamte Case Study schreiben
    save_training_info(case_study)

    env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=num_cpu,
        vec_env_cls=SubprocVecEnv,
    )
    eval_env = make_vec_env(
        lambda: TrainingEnv(constants),
        n_envs=1,
        vec_env_cls=SubprocVecEnv,
    )

    stop_callback = CustomStopCallback(
        max_no_improvement_evals=200,
        min_evals=200,
        verbose=1,
        param_label=f"CS{case_study_id}-{CaseStudy.coherence_time_str(coherence_time)}",
    )

    actual_eval_freq = max(1, 20_000 // num_cpu)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=actual_eval_freq,
        n_eval_episodes=70,
        callback_after_eval=stop_callback,
        best_model_save_path=ct_folder,  # → best_model.zip landet hier
        verbose=1,
        deterministic=case_study.deterministic,
    )

    best_model_path = os.path.join(ct_folder, "best_model.zip")

    if os.path.exists(best_model_path):
        print(f"Lade existierendes Modell: {best_model_path}")
        model = PPO.load(best_model_path, env=env, device="cpu")
        info = load_training_info(case_study_id)
        if info:
            print("📂 Vorhandene Case Study – gespeicherte Trainingsinfos:")
            print_training_info(info)
        else:
            print("⚠️  Keine training_info.json gefunden (altes Modell).")
    else:
        print(
            f"Starte neues Training: Case Study {case_study_id}, T_coh={coherence_time}"
        )
        h = case_study.hyperparams
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=dict(pi=h.pi_layers, vf=h.vf_layers),
                activation_fn=torch.nn.Tanh,
            ),
            n_steps=h.n_steps,
            batch_size=h.batch_size,
            n_epochs=h.n_epochs,
            learning_rate=h.learning_rate,
            gamma=h.gamma,
            gae_lambda=h.gae_lambda,
            ent_coef=h.ent_coef,
            clip_range=h.clip_range,
            vf_coef=h.vf_coef,
            max_grad_norm=h.max_grad_norm,
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
        env.close()
        eval_env.close()

    end_model_path = os.path.join(ct_folder, "end_model.zip")
    model.save(end_model_path)
    print(f"End-Modell gespeichert: {end_model_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main():
    # ──────────────────────────────────────────────────────────────────────────
    # Hier konfigurierst du, welche Case Study und welche Kohärenzzeiten
    # auf DIESEM Gerät trainiert werden sollen.
    #
    # Beispiel PC:    COHERENCE_TIMES = [0.01, 0.05]
    # Beispiel Laptop: COHERENCE_TIMES = [0.09]
    # ──────────────────────────────────────────────────────────────────────────
    CASE_STUDY_ID = 15
    COHERENCE_TIMES = CASE_STUDIES[CASE_STUDY_ID].coherence_times
    NUM_CORES_PER_RUN = 6

    for coherence_time in COHERENCE_TIMES:
        print(f"\n{'═' * 55}")
        print(
            f"  Case Study {CASE_STUDY_ID} | T_coh = {coherence_time} | {NUM_CORES_PER_RUN} Cores"
        )
        print(f"{'═' * 55}")
        train(CASE_STUDY_ID, coherence_time, NUM_CORES_PER_RUN)


if __name__ == "__main__":
    main()
