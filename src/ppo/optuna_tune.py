"""
optuna_tune.py
──────────────
Sucht optimale PPO-Hyperparameter mit Optuna.

Optimierte Parameter
    n_steps, batch_size, n_epochs, learning_rate, gae_lambda, ent_coef

Alle anderen Parameter (gamma, clip_range, vf_coef, max_grad_norm,
net_arch) werden aus der gewählten Case Study übernommen.

Verwendung
    python -m ppo.optuna_tune
oder mit Argumenten (optional):
    python -m ppo.optuna_tune --case-study 1 --coherence-time 0.05 \
                              --trials 50 --eval-episodes 30 --timesteps 500000
"""

import argparse
import os

import optuna
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from ppo.case_studies import CASE_STUDIES
from ppo.custom_env import TrainingEnv

# ─── Konfiguration ────────────────────────────────────────────────────────────

DEFAULT_CASE_STUDY_ID = 13
DEFAULT_COHERENCE_TIME = 0.007  # eine repräsentative Kohärenzzeit zum Tunen
DEFAULT_N_TRIALS = 30  # Anzahl Optuna-Trials
DEFAULT_EVAL_EPISODES = 50  # Episoden pro Evaluation im Trial
DEFAULT_TIMESTEPS = 400_000  # Trainingsschritte pro Trial
DEFAULT_NUM_CPU = 12  # parallele Envs während des Tunings
STUDY_NAME = "ppo_hyperopt"
STORAGE_URL = None  # z.B. "sqlite:///optuna_study.db" für Persistenz


# ─── Objective ────────────────────────────────────────────────────────────────


def make_objective(
    case_study_id: int,
    coherence_time: float,
    n_eval_episodes: int,
    total_timesteps: int,
    num_cpu: int,
):
    """Gibt eine Optuna-Objective-Funktion zurück (Closure über die Konfig)."""

    case_study = CASE_STUDIES[case_study_id]
    constants = case_study.make_constants(coherence_time)
    h = case_study.hyperparams  # Basiswerte für fixe Parameter

    best_reward = float("-inf")
    out_dir = "results/optuna"
    os.makedirs(out_dir, exist_ok=True)
    best_model_path = os.path.join(
        out_dir,
        f"best_model_cs{case_study_id}_T{str(coherence_time).replace('.', '_')}.zip",
    )

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_reward
        # ── Suchraum ──────────────────────────────────────────────────────────
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048, 4096])
        # Suggest from a fixed list so Optuna's CategoricalDistribution stays
        # consistent across trials (dynamic value spaces are not supported).
        # We then snap down to the largest value that divides total_steps.
        _BATCH_CHOICES = [32, 64, 128, 256, 512, 1024]
        total_steps = n_steps * num_cpu
        batch_size_hint = trial.suggest_categorical("batch_size", _BATCH_CHOICES)
        # Find the largest valid divisor ≤ the suggested hint
        valid = [
            b for b in _BATCH_CHOICES if total_steps % b == 0 and b <= batch_size_hint
        ]
        batch_size = valid[-1] if valid else min(_BATCH_CHOICES[0], total_steps)

        n_epochs = trial.suggest_int("n_epochs", 3, 15)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 1.0)
        ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)

        # ── Envs aufbauen ─────────────────────────────────────────────────────
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

        # ── Modell ────────────────────────────────────────────────────────────
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(
                net_arch=dict(pi=h.pi_layers, vf=h.vf_layers),
                activation_fn=torch.nn.Tanh,
            ),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            gamma=h.gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            clip_range=h.clip_range,
            vf_coef=h.vf_coef,
            max_grad_norm=h.max_grad_norm,
            device="cpu",
            verbose=0,
        )

        try:
            model.learn(total_timesteps=total_timesteps)
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=n_eval_episodes,
                deterministic=case_study.deterministic,
            )
            if isinstance(mean_reward, float) and mean_reward > best_reward:
                best_reward = mean_reward
                model.save(best_model_path)
                print(
                    f"[Trial {trial.number}] Neues bestes Modell gespeichert ({mean_reward:.4f} > {best_reward:.4f})"
                )
        except Exception as e:
            print(f"[Trial {trial.number}] Fehler: {e}")
            mean_reward = float("-inf")
        finally:
            env.close()
            eval_env.close()

        print(
            f"[Trial {trial.number:3d}]  "
            f"n_steps={n_steps:4d}  batch={batch_size:4d}  "
            f"epochs={n_epochs:2d}  lr={learning_rate:.2e}  "
            f"gae={gae_lambda:.3f}  ent={ent_coef:.2e}  "
            f"→ mean_reward={mean_reward:.4f}  std={std_reward:.4f}"
        )
        return mean_reward

    return objective


# ─── Study ────────────────────────────────────────────────────────────────────


def run_study(
    case_study_id: int,
    coherence_time: float,
    n_trials: int,
    n_eval_episodes: int,
    total_timesteps: int,
    num_cpu: int,
) -> None:
    # Optuna-Logs etwas leiser schalten
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage = STORAGE_URL  # None → in-memory (nicht persistent)

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )

    objective = make_objective(
        case_study_id=case_study_id,
        coherence_time=coherence_time,
        n_eval_episodes=n_eval_episodes,
        total_timesteps=total_timesteps,
        num_cpu=num_cpu,
    )

    print(f"\n{'═' * 60}")
    print(f"  Optuna Hyperparameter-Suche")
    print(f"  Case Study : {case_study_id}")
    print(f"  T_coh      : {coherence_time}")
    print(f"  Trials     : {n_trials}")
    print(f"  Timesteps  : {total_timesteps:,}")
    print(f"  Eval-Eps.  : {n_eval_episodes}")
    print(f"  CPUs       : {num_cpu}")
    print(f"{'═' * 60}\n")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # ── Ergebnis ──────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'─' * 60}")
    print(f"  ✅ Bester Trial #{best.number}  –  mean_reward = {best.value:.4f}")
    print(f"{'─' * 60}")
    for k, v in best.params.items():
        print(f"    {k:20s}: {v}")
    print(f"{'─' * 60}\n")

    # ── Als HyperparamsTuple-Snippet ausgeben ─────────────────────────────────
    p = best.params
    cs = CASE_STUDIES[case_study_id]
    h = cs.hyperparams
    print("  📋 HyperparamsTuple-Snippet (zum Eintragen in case_studies.py):")
    print(f"  HyperparamsTuple(")
    print(f"      n_steps       = {p['n_steps']},")
    print(f"      batch_size    = {p['batch_size']},")
    print(f"      n_epochs      = {p['n_epochs']},")
    print(f"      learning_rate = {p['learning_rate']:.6e},")
    print(f"      gamma         = {h.gamma},")
    print(f"      gae_lambda    = {p['gae_lambda']:.6f},")
    print(f"      ent_coef      = {p['ent_coef']:.6e},")
    print(f"      clip_range    = {h.clip_range},")
    print(f"      vf_coef       = {h.vf_coef},")
    print(f"      max_grad_norm = {h.max_grad_norm},")
    print(f"      pi_layers     = {h.pi_layers},")
    print(f"      vf_layers     = {h.vf_layers},")
    print(f"  )")

    # ── Optional: Ergebnisse als JSON speichern ────────────────────────────────
    import json

    out_dir = "results/optuna"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"best_params_cs{case_study_id}_T{str(coherence_time).replace('.', '_')}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "case_study_id": case_study_id,
                "coherence_time": coherence_time,
                "best_trial": best.number,
                "best_value": best.value,
                "best_params": best.params,
            },
            f,
            indent=2,
        )
    print(f"  💾 Ergebnisse gespeichert: {out_path}")


# ─── Entry Point ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter-Tuning für PPO (purify)"
    )
    parser.add_argument(
        "--case-study",
        type=int,
        default=DEFAULT_CASE_STUDY_ID,
        help=f"Case Study ID (default: {DEFAULT_CASE_STUDY_ID})",
    )
    parser.add_argument(
        "--coherence-time",
        type=float,
        default=DEFAULT_COHERENCE_TIME,
        help=f"Kohärenzzeit zum Tunen (default: {DEFAULT_COHERENCE_TIME})",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Anzahl Optuna-Trials (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Eval-Episoden pro Trial (default: {DEFAULT_EVAL_EPISODES})",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Trainingsschritte pro Trial (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--num-cpu",
        type=int,
        default=DEFAULT_NUM_CPU,
        help=f"Parallele Envs (default: {DEFAULT_NUM_CPU})",
    )
    args = parser.parse_args()

    run_study(
        case_study_id=args.case_study,
        coherence_time=args.coherence_time,
        n_trials=args.trials,
        n_eval_episodes=args.eval_episodes,
        total_timesteps=args.timesteps,
        num_cpu=args.num_cpu,
    )


if __name__ == "__main__":
    main()
