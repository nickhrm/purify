"""
evaluate_replace_vs_coherence_time.py
──────────────────────────────────────
Case-Study-Analyse: Ab welcher wait_time_sensitivity wird
KEEP BEST (= REPLACE) zur dominanten Aktion?

Hintergrund
───────────
`waiting_time_sensitivity` (ConstantsTuple) multipliziert `time_alive`
des gespeicherten Qubits im Speicher:

    f_mem ∝ exp(-(time_alive * waiting_time_sensitivity) / coherence_time)

Ein höherer Wert lässt den Speicher schneller altern, d.h. f_mem fällt schneller.
→ Das Modell bekommt im Laufe der Zeit niedrigere f_mem-Werte präsentiert.

Methode
───────
1. Für eine feste (case_study_id, coherence_time) wird das best_model geladen.
2. waiting_time_sensitivity wird über SENSITIVITY_VALUES gesweept.
3. Pro Sensitivity-Wert werden N_EPISODES echte Rollouts gefahren.
   Das Env nutzt den gesweepten Wert über ein modifiziertes ConstantsTuple.
4. REPLACE-Anteil und p(REPLACE) werden pro Wert gemessen.
5. Der erste Sensitivity-Wert, bei dem REPLACE die häufigste Aktion ist,
   gilt als Threshold.

Output
──────
results/case_study_{id}/replace_vs_wait_sensitivity_{coherence_time_str}.csv
  Spalten: waiting_time_sensitivity, replace_pct, dominant_action, replace_is_dominant

Konfiguration (unten anpassen)
──────────────────────────────
  CASE_STUDY_ID       – welche Case Study
  COHERENCE_TIME      – für welche Coherence Time das Modell geladen wird
  SENSITIVITY_VALUES  – zu sweepende waiting_time_sensitivity-Werte
  N_EPISODES          – Rollouts pro Sensitivity-Wert
  MODEL_FILE          – welches Modell geladen wird
"""

import os
import csv
from collections import Counter

import numpy as np
from stable_baselines3 import PPO

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent
from ppo.trainer import coherence_time_folder, case_study_root

# ─── Konfiguration ────────────────────────────────────────────────────────────

CASE_STUDY_ID: int = 1

# Die Coherence Time, für die das trainierte Modell geladen wird
COHERENCE_TIME: float = 0.05

# waiting_time_sensitivity-Werte, die gesweept werden
SENSITIVITY_VALUES: list[float] = [
    1, 5, 15, 20
]

# Rollouts pro Sensitivity-Wert
N_EPISODES: int = 1500

# Welches Modell geladen wird
MODEL_FILE: str = "best_model.zip"

# ─── Kern-Simulation ──────────────────────────────────────────────────────────


def run_rollouts(
    model_path: str,
    base_constants,
    sensitivity: float,
    n_episodes: int,
) -> tuple[Counter, float]:
    """
    Führt `n_episodes` Rollouts mit einem modifizierten ConstantsTuple durch,
    in dem waiting_time_sensitivity auf `sensitivity` gesetzt ist.

    Gibt (action_counts, avg_reward) zurück.
    """
    constants = base_constants._replace(waiting_time_sensitivity=sensitivity)
    env = TrainingEnv(constants)
    agent = SB3Agent(model_path, env)

    action_counts: Counter = Counter()
    total_reward = 0.0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            action_name = constants.actions[action]
            action_counts[action_name] += 1
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

    env.close()
    return action_counts, total_reward / n_episodes


# ─── Speichern ────────────────────────────────────────────────────────────────


def save_results_csv(case_study_id: int, coherence_time: float, rows: list[dict]) -> None:
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    ct_str = CaseStudy.coherence_time_str(coherence_time)
    path = os.path.join(root, f"replace_vs_wait_sensitivity_{ct_str}.csv")

    fieldnames = [
        "waiting_time_sensitivity",
        "replace_pct",
        "dominant_action",
        "replace_is_dominant",
        "avg_reward",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  💾  Gespeichert: {path}")


# ─── Haupt-Evaluierung ────────────────────────────────────────────────────────


def evaluate_replace_vs_wait_sensitivity() -> None:
    case_study = CASE_STUDIES[CASE_STUDY_ID]
    actions = case_study.constants.actions

    ct_folder  = coherence_time_folder(CASE_STUDY_ID, COHERENCE_TIME)
    model_path = os.path.join(ct_folder, MODEL_FILE)
    ct_str     = CaseStudy.coherence_time_str(COHERENCE_TIME)

    if not os.path.exists(model_path):
        print(f"❌  Modell nicht gefunden: {model_path}")
        return

    base_constants = case_study.make_constants(COHERENCE_TIME)

    print(f"\n{'═' * 72}")
    print(f"  REPLACE vs. waiting_time_sensitivity  |  CS {CASE_STUDY_ID}  |  T_coh={COHERENCE_TIME}")
    print(f"{'═' * 72}\n")
    print(f"  {'sensitivity':>12}  {'p(REPLACE)%':>12}  {'Dominant':>16}  {'dom.':>6}  {'avg_rew':>8}")
    print(f"  {'-' * 62}")

    all_rows: list[dict] = []
    threshold_sensitivity: float | None = None

    for sensitivity in sorted(SENSITIVITY_VALUES):
        action_counts, avg_reward = run_rollouts(
            model_path, base_constants, sensitivity, N_EPISODES
        )

        total = sum(action_counts.values())
        replace_key = next(
            (k for k in action_counts if str(k) == "Action.REPLACE"), None
        )
        replace_count = action_counts[replace_key] if replace_key is not None else 0
        replace_pct = (replace_count / total * 100) if total > 0 else 0.0

        # Dominante Aktion = häufigste
        if action_counts:
            dominant_key = max(action_counts, key=lambda k: action_counts[k])
            dominant_name = str(dominant_key)
        else:
            dominant_name = "None"
        
        replace_is_dominant = dominant_name == "Action.REPLACE"

        if replace_is_dominant and threshold_sensitivity is None:
            threshold_sensitivity = sensitivity

        dom_marker = " ◀ THRESHOLD" if (replace_is_dominant and threshold_sensitivity == sensitivity) else ""
        print(
            f"  {sensitivity:>12.2f}  {replace_pct:>11.1f}%  "
            f"{dominant_name:>16}  {'✓' if replace_is_dominant else '✗':>6}  "
            f"{avg_reward:>8.4f}{dom_marker}"
        )

        all_rows.append({
            "waiting_time_sensitivity": sensitivity,
            "replace_pct":             round(replace_pct, 3),
            "dominant_action":         dominant_name,
            "replace_is_dominant":     replace_is_dominant,
            "avg_reward":              round(avg_reward, 6),
        })

    print(f"  {'-' * 62}")

    if threshold_sensitivity is not None:
        print(
            f"\n  🎯  THRESHOLD: REPLACE wird erstmals dominant bei "
            f"waiting_time_sensitivity = {threshold_sensitivity}"
        )
    else:
        print("\n  ℹ️   REPLACE ist für keinen der getesteten Sensitivity-Werte dominant.")

    save_results_csv(CASE_STUDY_ID, COHERENCE_TIME, all_rows)
    print(f"\n  ✅  Evaluation abgeschlossen.\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_replace_vs_wait_sensitivity()
