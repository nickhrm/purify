"""
training2.py
─────────────────────────────────────────────────────────────────────────────
Verbesserter Trainings-Workflow gegenüber trainer.py:

1. OPTUNA-FIRST
   Bevor das eigentliche Training startet, wird automatisch eine kompakte
   Optuna-Suche ausgeführt (oder ein gecachtes Ergebnis geladen).
   → Die besten Hyperparameter werden direkt verwendet, kein manuelles
     Übertragen in case_studies.py nötig.

2. MULTI-RUN (best-of-N)
   Das Training wird N_RUNS-mal mit denselben Hyperparametern neu gestartet.
   Am Ende wird der Run mit dem besten evaluierten mean_reward als
   ``best_model.zip`` gespeichert.
   → Ein schlechter Zufalls-Seed ruiniert nicht das ganze Ergebnis.

3. SCHNELLES EARLY-STOPPING
   - eval_freq   : 10 000 Schritte (statt 20 000)
   - no_improve  : 80 Evals       (statt 400)
   - min_evals   : 50             (statt 200)
   - max_steps   : 8 000 000      (statt 30 000 000)
   → Ein nicht-konvergierender Lauf wird ~10× schneller abgebrochen.

Verwendung
──────────
    python -m ppo.training2        # konfigurierbar im main()-Block unten
"""

import json
import os
import shutil
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.custom_stop_callback import CustomStopCallback
from ppo.hyperparams_tuple import HyperparamsTuple
from ppo.trainer import (
    coherence_time_folder,
    save_training_info,
)

# ─── Konfiguration ────────────────────────────────────────────────────────────

# Anzahl der Trainings-Wiederholungen pro Kohärenzzeit
N_RUNS: int = 3

# Optuna-Suche
OPTUNA_TRIALS: int = 30  # Trials (weniger als optuna_tune.py default=60)
OPTUNA_TIMESTEPS: int = 300_000  # Schritte pro Optuna-Trial
FORCE_TUNE: bool = False  # True: Optuna auch dann ausführen, wenn JSON existiert

# Trainings-Limits pro Run
MAX_TRAIN_STEPS: int = 8_000_000  # Maximale Gesamtschritte (vorher 30M)

# Early-Stopping (aggressiver als trainer.py)
EVAL_FREQ: int = 10_000  # Wie oft evaluiert wird (in Env-Schritten)
NO_IMPROVE_EVALS: int = 80  # Geduld: so viele schlechte Evals → Abbruch
MIN_EVALS: int = 50  # Warmup: Early-Stopping startet erst danach
N_EVAL_EPISODES: int = 70  # Episoden pro Evaluation


# ─── Optuna-Integration ───────────────────────────────────────────────────────


def _optuna_json_path(case_study_id: int, coherence_time: float) -> str:
    """Pfad zur gecachten Optuna-JSON-Datei."""
    ct_key = str(coherence_time).replace(".", "_")
    return os.path.join("results", "optuna", f"best_params_cs{case_study_id}_T{ct_key}.json")


def _hyperparams_from_json(json_path: str, case_study_id: int) -> HyperparamsTuple:
    """Liest Optuna-JSON und baut daraus ein HyperparamsTuple (fixe Werte aus CaseStudy)."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    params = data["best_params"]
    h_base = CASE_STUDIES[case_study_id].hyperparams  # feste Werte (gamma, clip_range, …)

    # batch_size: Optuna speichert den „hint", wir müssen sicherstellen, dass er gültig ist
    num_cpu_hint = data.get("num_cpu", 6)
    n_steps = int(params["n_steps"])
    total_steps = n_steps * num_cpu_hint
    batch_hint = int(params["batch_size"])
    _BATCH_CHOICES = [32, 64, 128, 256, 512, 1024]
    valid = [b for b in _BATCH_CHOICES if total_steps % b == 0 and b <= batch_hint]
    batch_size = valid[-1] if valid else min(_BATCH_CHOICES[0], total_steps)

    return HyperparamsTuple(
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=int(params["n_epochs"]),
        learning_rate=float(params["learning_rate"]),
        gamma=h_base.gamma,
        gae_lambda=float(params["gae_lambda"]),
        ent_coef=float(params["ent_coef"]),
        clip_range=h_base.clip_range,
        vf_coef=h_base.vf_coef,
        max_grad_norm=h_base.max_grad_norm,
        pi_layers=h_base.pi_layers,
        vf_layers=h_base.vf_layers,
    )


def load_or_tune_hyperparams(
    case_study_id: int,
    coherence_time: float,
    num_cpu: int,
    force_tune: bool = FORCE_TUNE,
    n_trials: int = OPTUNA_TRIALS,
    timesteps: int = OPTUNA_TIMESTEPS,
) -> HyperparamsTuple:
    """
    Gibt HyperparamsTuple zurück:
      - Aus gecachter Optuna-JSON (falls vorhanden und force_tune=False)
      - Nach einer frischen Optuna-Suche (und Speicherung in JSON)
    """
    json_path = _optuna_json_path(case_study_id, coherence_time)

    if not force_tune and os.path.exists(json_path):
        print(f"\n  📂 Lade gecachte Optuna-Ergebnisse: {json_path}")
        hp = _hyperparams_from_json(json_path, case_study_id)
        print(
            f"     n_steps={hp.n_steps}  batch={hp.batch_size}  "
            f"epochs={hp.n_epochs}  lr={hp.learning_rate:.2e}  "
            f"gae={hp.gae_lambda:.4f}  ent={hp.ent_coef:.2e}"
        )
        return hp

    # ── Frische Optuna-Suche ──────────────────────────────────────────────────
    print(f"\n  🔍 Starte Optuna-Suche ({n_trials} Trials × {timesteps:,} Steps) …")
    from ppo.optuna_tune import run_study  # lokaler Import um Kreisimport zu vermeiden

    run_study(
        case_study_id=case_study_id,
        coherence_time=coherence_time,
        n_trials=n_trials,
        n_eval_episodes=30,
        total_timesteps=timesteps,
        num_cpu=num_cpu,
    )
    # run_study() speichert die JSON selbst (→ results/optuna/…)
    hp = _hyperparams_from_json(json_path, case_study_id)
    return hp


# ─── Einzelner Trainingslauf ──────────────────────────────────────────────────


def train_single_run(
    case_study_id: int,
    coherence_time: float,
    hyperparams: HyperparamsTuple,
    num_cpu: int,
    run_id: int,
    save_dir: str,
) -> tuple[PPO, float]:
    """
    Führt einen einzelnen Trainingslauf durch.

    Returns
    -------
    model       : das trainierte PPO-Modell
    mean_reward : evaluierter mean_reward (aus N_EVAL_EPISODES Episoden)
    """
    label = f"CS{case_study_id}-T{CaseStudy.coherence_time_str(coherence_time)}-Run{run_id}"
    print(f"\n  {'─' * 52}")
    print(f"  ▶ {label}")
    print(f"  {'─' * 52}")

    case_study = CASE_STUDIES[case_study_id]
    constants = case_study.make_constants(coherence_time)

    log_dir = os.path.join(
        "logs",
        f"case_study_{case_study_id}",
        CaseStudy.coherence_time_str(coherence_time),
        f"run{run_id}",
    )
    os.makedirs(log_dir, exist_ok=True)

    # Temp-Ordner für best_model dieses Runs (EvalCallback braucht einen Pfad)
    run_best_dir = os.path.join(save_dir, f"_run{run_id}_tmp")
    os.makedirs(run_best_dir, exist_ok=True)

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
        max_no_improvement_evals=NO_IMPROVE_EVALS,
        min_evals=MIN_EVALS,
        verbose=1,
        param_label=label,
    )

    actual_eval_freq = max(1, EVAL_FREQ // num_cpu)
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=actual_eval_freq,
        n_eval_episodes=N_EVAL_EPISODES,
        callback_after_eval=stop_callback,
        best_model_save_path=run_best_dir,
        verbose=1,
        deterministic=case_study.deterministic,
    )

    h = hyperparams
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

    try:
        model.learn(
            total_timesteps=MAX_TRAIN_STEPS,
            reset_num_timesteps=True,
            callback=eval_callback,
        )
    except KeyboardInterrupt:
        print(f"  ⚠️  {label}: Training manuell unterbrochen.")
    finally:
        env.close()
        eval_env.close()

    # Lade das beste Modell dieses Runs für die finale Evaluation
    best_run_path = os.path.join(run_best_dir, "best_model.zip")
    if os.path.exists(best_run_path):
        eval_env2 = make_vec_env(
            lambda: TrainingEnv(constants),
            n_envs=1,
            vec_env_cls=SubprocVecEnv,
        )
        best_model = PPO.load(best_run_path, env=eval_env2, device="cpu")
        mean_reward, std_reward = evaluate_policy(
            best_model,
            eval_env2,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=case_study.deterministic,
        )
        eval_env2.close()
        print(f"  ✅ {label}: mean_reward = {mean_reward:.4f}  (std = {std_reward:.4f})")
        return best_model, float(mean_reward)
    else:
        # Kein best_model (Training lief nie long genug) → nimm das end-model
        print(f"  ⚠️  {label}: Kein best_model gefunden, nutze end-model für Evaluation.")
        eval_env2 = make_vec_env(
            lambda: TrainingEnv(constants),
            n_envs=1,
            vec_env_cls=SubprocVecEnv,
        )
        model.set_env(eval_env2)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env2,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=case_study.deterministic,
        )
        eval_env2.close()
        print(f"  ⚠️  {label}: mean_reward = {mean_reward:.4f}  (std = {std_reward:.4f})")
        return model, float(mean_reward)


# ─── Haupt-Trainingsfunktion ──────────────────────────────────────────────────


def train(
    case_study_id: int,
    coherence_time: float,
    num_cpu: int,
    n_runs: int = N_RUNS,
    force_tune: bool = FORCE_TUNE,
) -> None:
    """
    Führt den vollständigen training2-Workflow für eine Kohärenzzeit durch:
      1. Optuna: Hole beste Hyperparameter (gecacht oder frisch)
      2. N Trainingsläufe
      3. Bestes Modell → best_model.zip
      4. Zusammenfassung → training2_info.json
    """
    ct_folder = coherence_time_folder(case_study_id, coherence_time)
    os.makedirs(ct_folder, exist_ok=True)

    print(f"\n{'═' * 58}")
    print(f"  training2  |  Case Study {case_study_id}  |  T_coh = {coherence_time}")
    print(f"{'═' * 58}")

    # ── 1. Hyperparameter ─────────────────────────────────────────────────────
    hyperparams = load_or_tune_hyperparams(
        case_study_id=case_study_id,
        coherence_time=coherence_time,
        num_cpu=num_cpu,
        force_tune=force_tune,
    )

    # einmalig training_info.json schreiben (aus trainer.py wiederverwendet)
    save_training_info(CASE_STUDIES[case_study_id])

    # ── 2. Multi-Run ──────────────────────────────────────────────────────────
    run_results: list[tuple[int, float]] = []  # (run_id, mean_reward)
    best_reward = float("-inf")
    best_run_id = -1

    for run_id in range(1, n_runs + 1):
        model, mean_reward = train_single_run(
            case_study_id=case_study_id,
            coherence_time=coherence_time,
            hyperparams=hyperparams,
            num_cpu=num_cpu,
            run_id=run_id,
            save_dir=ct_folder,
        )
        run_results.append((run_id, mean_reward))

        if mean_reward > best_reward:
            best_reward = mean_reward
            best_run_id = run_id
            # Speichere bestes Modell sofort (überschreibt ggf. vorherige)
            tmp_best = os.path.join(ct_folder, f"_run{run_id}_tmp", "best_model.zip")
            final_best = os.path.join(ct_folder, "best_model.zip")
            if os.path.exists(tmp_best):
                shutil.copy2(tmp_best, final_best)
            else:
                model.save(final_best)
            print(f"\n  🏆 Neues bestes Modell: Run {run_id}  →  {mean_reward:.4f}")

    # ── 3. Aufräumen ──────────────────────────────────────────────────────────
    for run_id in range(1, n_runs + 1):
        tmp_dir = os.path.join(ct_folder, f"_run{run_id}_tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    # ── 4. Zusammenfassung ────────────────────────────────────────────────────
    print(f"\n{'─' * 58}")
    print(f"  Zusammenfassung  |  CS {case_study_id}  |  T_coh = {coherence_time}")
    print(f"{'─' * 58}")
    for run_id, reward in run_results:
        marker = " ← BEST" if run_id == best_run_id else ""
        print(f"    Run {run_id}: mean_reward = {reward:.4f}{marker}")
    print(f"{'─' * 58}")
    print(f"  💾 best_model.zip gespeichert in: {ct_folder}")

    # JSON-Zusammenfassung
    info = {
        "case_study_id": case_study_id,
        "coherence_time": coherence_time,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "workflow": "training2 (Optuna-first, best-of-N)",
        "config": {
            "n_runs": n_runs,
            "max_train_steps": MAX_TRAIN_STEPS,
            "eval_freq": EVAL_FREQ,
            "no_improve_evals": NO_IMPROVE_EVALS,
            "min_evals": MIN_EVALS,
            "n_eval_episodes": N_EVAL_EPISODES,
        },
        "hyperparams": hyperparams.to_dict(),
        "runs": [
            {"run_id": rid, "mean_reward": r, "is_best": rid == best_run_id}
            for rid, r in run_results
        ],
        "best_run_id": best_run_id,
        "best_reward": best_reward,
    }
    info_path = os.path.join(ct_folder, "training2_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  📋 training2_info.json → {info_path}")
    print(f"{'═' * 58}\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main() -> None:
    # ──────────────────────────────────────────────────────────────────────────
    # Konfiguration – hier anpassen:
    # ──────────────────────────────────────────────────────────────────────────
    CASE_STUDY_ID = 11
    COHERENCE_TIMES = CASE_STUDIES[CASE_STUDY_ID].coherence_times
    NUM_CORES = 6
    N_RUNS_CFG = N_RUNS  # Anzahl Trainings-Wiederholungen
    FORCE_TUNE_CFG = FORCE_TUNE  # Optuna immer neu ausführen?
    # ──────────────────────────────────────────────────────────────────────────

    for coherence_time in COHERENCE_TIMES:
        train(
            case_study_id=CASE_STUDY_ID,
            coherence_time=coherence_time,
            num_cpu=NUM_CORES,
            n_runs=N_RUNS_CFG,
            force_tune=FORCE_TUNE_CFG,
        )


if __name__ == "__main__":
    main()
