"""
evaluate_replace_vs_coherence_time.py
──────────────────────────────────────
Case-Study-Analyse: Wie beeinflusst waiting_time_sensitivity (κ)
das Verhalten des Modells?

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
4. Gemessen wird pro κ-Wert:
   - Volle Aktionsverteilung (% pro Aktion)
   - Durchschnittliche f_mem an Entscheidungspunkten
   - Teleportation Fidelity (nur bei terminierten Episoden)
   - Avg Reward (über alle Episoden inkl. truncated)
   - Threshold: erster κ-Wert, bei dem REPLACE dominant wird

Output
──────
results/case_study_{id}/replace_vs_wait_sensitivity_{coherence_time_str}.csv

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
from dataclasses import dataclass

import numpy as np
from stable_baselines3 import PPO

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent
from ppo.trainer import coherence_time_folder, case_study_root

# ─── Konfiguration ────────────────────────────────────────────────────────────

CASE_STUDY_ID: int = 15

# Die Coherence Time, für die das trainierte Modell geladen wird
COHERENCE_TIME: float = 0.05

# waiting_time_sensitivity-Werte, die gesweept werden
SENSITIVITY_VALUES: list[float] = [1, 5, 15, 20]

# Rollouts pro Sensitivity-Wert
N_EPISODES: int = 1500

# Welches Modell geladen wird
MODEL_FILE: str = "best_model.zip"


# ─── Ergebnis-Container ──────────────────────────────────────────────────────


@dataclass
class RolloutResult:
    """Alle Metriken eines κ-Sweeps."""

    action_counts: Counter
    action_pcts: dict[str, float]  # Aktion → Prozent
    avg_reward: float  # Über alle Episoden (inkl. truncated)
    avg_teleportation_fidelity: float  # Nur terminierte Episoden
    n_terminated: int  # Anzahl terminierter Episoden
    avg_f_mem_at_decision: float  # Durchschn. f_mem an Entscheidungspunkten


# ─── Kern-Simulation ──────────────────────────────────────────────────────────


def run_rollouts(
    model_path: str,
    base_constants,
    sensitivity: float,
    n_episodes: int,
) -> RolloutResult:
    """
    Führt `n_episodes` Rollouts mit einem modifizierten ConstantsTuple durch,
    in dem waiting_time_sensitivity auf `sensitivity` gesetzt ist.

    Trackt:
      - Aktionsverteilung (alle Aktionen, nicht nur REPLACE)
      - f_mem an jedem Entscheidungspunkt
      - Teleportation Fidelity (nur terminierte Episoden)
      - Reward (alle Episoden)
    """
    constants = base_constants._replace(waiting_time_sensitivity=sensitivity)
    env = TrainingEnv(constants)
    agent = SB3Agent(model_path, env)

    action_counts: Counter = Counter()
    total_reward = 0.0
    teleportation_fidelities: list[float] = []
    f_mem_at_decisions: list[float] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        # f_mem beim ersten Entscheidungspunkt der Episode
        f_mem_at_decisions.append(info["f_mem"])

        done = False
        while not done:
            action = agent.predict(obs)
            action_name = constants.actions[action]
            action_counts[action_name] += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated:
                # reward == teleportation_fidelity bei Terminierung
                teleportation_fidelities.append(reward)
                done = True
            elif truncated:
                done = True
            else:
                # Nächster Entscheidungspunkt innerhalb der Episode
                f_mem_at_decisions.append(info["f_mem"])

    env.close()

    # Aktionsverteilung berechnen
    total_actions = sum(action_counts.values())
    action_pcts = {}
    for action in constants.actions:
        count = action_counts.get(action, 0)
        action_pcts[action.name] = round(
            (count / total_actions * 100) if total_actions > 0 else 0.0, 3
        )

    n_terminated = len(teleportation_fidelities)

    return RolloutResult(
        action_counts=action_counts,
        action_pcts=action_pcts,
        avg_reward=total_reward / n_episodes,
        avg_teleportation_fidelity=float(np.mean(teleportation_fidelities))
        if teleportation_fidelities
        else 0.0,
        n_terminated=n_terminated,
        avg_f_mem_at_decision=float(np.mean(f_mem_at_decisions))
        if f_mem_at_decisions
        else 0.0,
    )


# ─── Speichern ────────────────────────────────────────────────────────────────


def save_results_csv(
    case_study_id: int,
    coherence_time: float,
    rows: list[dict],
    action_names: list[str],
) -> None:
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    ct_str = CaseStudy.coherence_time_str(coherence_time)
    path = os.path.join(root, f"replace_vs_wait_sensitivity_{ct_str}.csv")

    # Dynamische Spalten für Aktionsverteilung
    action_cols = [f"pct_{name}" for name in action_names]
    fieldnames = [
        "waiting_time_sensitivity",
        *action_cols,
        "dominant_action",
        "replace_is_dominant",
        "avg_f_mem_at_decision",
        "avg_teleportation_fidelity",
        "n_terminated",
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
    action_names = [a.name for a in actions]

    ct_folder = coherence_time_folder(CASE_STUDY_ID, COHERENCE_TIME)
    model_path = os.path.join(ct_folder, MODEL_FILE)

    if not os.path.exists(model_path):
        print(f"❌  Modell nicht gefunden: {model_path}")
        return

    base_constants = case_study.make_constants(COHERENCE_TIME)

    # ─── Header ───────────────────────────────────────────────────────────
    print(f"\n{'═' * 90}")
    print(f"  κ-Sweep  |  CS {CASE_STUDY_ID}  |  T_coh={COHERENCE_TIME}")
    print(f"{'═' * 90}\n")

    # Dynamischer Header basierend auf verfügbaren Aktionen
    action_headers = "  ".join(f"{name:>8}" for name in action_names)
    print(
        f"  {'κ':>6}  {action_headers}"
        f"  {'Dominant':>16}  {'avg_f_mem':>9}  {'avg_F_T':>8}  {'avg_rew':>8}"
    )
    print(f"  {'-' * (80 + len(action_names) * 10)}")

    all_rows: list[dict] = []
    threshold_sensitivity: float | None = None

    for sensitivity in sorted(SENSITIVITY_VALUES):
        result = run_rollouts(model_path, base_constants, sensitivity, N_EPISODES)

        # Dominante Aktion bestimmen
        if result.action_counts:
            dominant_key = max(
                result.action_counts, key=lambda k: result.action_counts[k]
            )
            dominant_name = dominant_key.name
        else:
            dominant_name = "None"

        replace_is_dominant = dominant_name == "REPLACE"

        if replace_is_dominant and threshold_sensitivity is None:
            threshold_sensitivity = sensitivity

        dom_marker = (
            " ◀ THRESHOLD"
            if (replace_is_dominant and threshold_sensitivity == sensitivity)
            else ""
        )

        # Ausgabe
        pct_strs = "  ".join(
            f"{result.action_pcts.get(n, 0.0):>7.1f}%" for n in action_names
        )
        print(
            f"  {sensitivity:>6.1f}  {pct_strs}"
            f"  {dominant_name:>16}  {result.avg_f_mem_at_decision:>9.4f}"
            f"  {result.avg_teleportation_fidelity:>8.4f}  {result.avg_reward:>8.4f}{dom_marker}"
        )

        # CSV-Zeile bauen
        row: dict = {"waiting_time_sensitivity": sensitivity}
        for name in action_names:
            row[f"pct_{name}"] = result.action_pcts.get(name, 0.0)
        row.update(
            {
                "dominant_action": dominant_name,
                "replace_is_dominant": replace_is_dominant,
                "avg_f_mem_at_decision": round(result.avg_f_mem_at_decision, 6),
                "avg_teleportation_fidelity": round(
                    result.avg_teleportation_fidelity, 6
                ),
                "n_terminated": result.n_terminated,
                "avg_reward": round(result.avg_reward, 6),
            }
        )
        all_rows.append(row)

    print(f"  {'-' * (80 + len(action_names) * 10)}")

    if threshold_sensitivity is not None:
        print(
            f"\n  🎯  THRESHOLD: REPLACE wird erstmals dominant bei "
            f"waiting_time_sensitivity = {threshold_sensitivity}"
        )
    else:
        print(
            "\n  ℹ️   REPLACE ist für keinen der getesteten Sensitivity-Werte dominant."
        )

    save_results_csv(CASE_STUDY_ID, COHERENCE_TIME, all_rows, action_names)
    print(f"\n  ✅  Evaluation abgeschlossen.\n")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate_replace_vs_wait_sensitivity()
