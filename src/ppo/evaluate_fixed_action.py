"""
evaluate_fixed_action.py
------------------------
Evaluiert eine Baseline-Strategie, bei der der Agent immer dieselbe Aktion waehlt.
Es koennen mehrere Aktionen angegeben werden – sie werden nacheinander evaluiert.

Verwendung:
    python -m ppo.evaluate_fixed_action

Konfiguration: Passe die Variablen im Abschnitt 'Konfiguration' unten an.
"""

import csv
import os

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import FixedActionAgent
from purify.my_enums import Action

# === Konfiguration ============================================================

# Die ID der Case Study (muss in CASE_STUDIES existieren).
CASE_STUDY_ID: int = 6

# Aktionen, die nacheinander evaluiert werden sollen.
FIXED_ACTIONS: list[Action] = [
    Action.REPLACE,
    Action.PROT_1,
    Action.PROT_2,
    Action.PROT_3,
]

# Anzahl der Episoden pro Kohaerenzzeit und Aktion.
N_EPISODES: int = 3000

# Eigene Kohaerenzzeiten – ueberschreibt die Zeiten aus der Case Study.
# Leer lassen ([]), um die Zeiten der jeweiligen Case Study zu verwenden.
COHERENCE_TIMES: list[float] = [
            0.001,
            0.002,
            0.003,
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.1,
]

# =============================================================================


def _results_path(case_study_id: int) -> str:
    return os.path.join("results", f"case_study_{case_study_id}", "fixed_action_evaluation.csv")


def _save_results(rows: list[dict], case_study_id: int) -> None:
    path = _results_path(case_study_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.isfile(path)

    with open(path, mode="a", newline="") as f:
        fieldnames = ["case_study_id", "coherence_time", "fixed_action", "avg_reward", "n_episodes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"  Ergebnisse gespeichert: {path}")


def _evaluate_single_action(
    case_study: CaseStudy,
    fixed_action: Action,
    t_cohs: list[float],
    n_episodes: int,
) -> list[dict]:
    """Evaluiert eine einzelne feste Aktion ueber alle Kohaerenzzeiten."""
    print(f"\n  --- Aktion: {fixed_action.name} ---")

    if fixed_action not in case_study.constants.actions:
        available = [a.name for a in case_study.constants.actions]
        raise ValueError(
            f"Aktion '{fixed_action.name}' ist in Case Study {case_study.id} nicht verfuegbar. "
            f"Verfuegbar: {available}"
        )

    agent = FixedActionAgent(fixed_action)
    rows: list[dict] = []

    for coherence_time in t_cohs:
        constants = case_study.make_constants(coherence_time)
        ct_str = CaseStudy.coherence_time_str(coherence_time)
        print(f"    T_coh = {coherence_time} ({ct_str}) ...", end="", flush=True)

        env = TrainingEnv(constants)
        total_reward = 0.0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                action = agent.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

        avg_reward = total_reward / n_episodes
        print(f"  avg_reward = {avg_reward:.4f}")

        rows.append(
            {
                "case_study_id": case_study.id,
                "coherence_time": coherence_time,
                "fixed_action": fixed_action.name,
                "avg_reward": avg_reward,
                "n_episodes": n_episodes,
            }
        )

    return rows


def evaluate_fixed_actions(
    case_study_id: int = CASE_STUDY_ID,
    fixed_actions: list[Action] | None = None,
    n_episodes: int = N_EPISODES,
    coherence_times: list[float] | None = None,
) -> None:
    """
    Evaluiert die Strategie 'immer fixed_action' fuer jede angegebene Aktion
    nacheinander.

    Parameters
    ----------
    case_study_id : int
        ID der Case Study (muss in CASE_STUDIES vorhanden sein).
    fixed_actions : list[Action] | None
        Liste der Aktionen, die nacheinander evaluiert werden. Falls None,
        wird FIXED_ACTIONS verwendet.
    n_episodes : int
        Anzahl der Episoden pro Kohaerenzzeit und Aktion.
    coherence_times : list[float] | None
        Eigene Kohaerenzzeiten. Falls None oder leer, werden die Zeiten
        der Case Study verwendet.
    """
    actions = fixed_actions if fixed_actions else FIXED_ACTIONS
    case_study: CaseStudy = CASE_STUDIES[case_study_id]
    t_cohs = coherence_times if coherence_times else case_study.coherence_times

    print(f"\n{'=' * 55}")
    print(f"  Fixed-Action-Evaluierung  -  Case Study {case_study_id}")
    print(f"  Aktionen       : {[a.name for a in actions]}")
    print(f"  Kohaerenzzeiten: {t_cohs}")
    print(f"  Episoden/T_coh : {n_episodes}")
    print(f"{'=' * 55}")

    all_rows: list[dict] = []
    for action in actions:
        rows = _evaluate_single_action(case_study, action, t_cohs, n_episodes)
        all_rows.extend(rows)

    _save_results(all_rows, case_study_id)
    print(f"\n  Fertig. {len(actions)} Aktion(en), {len(t_cohs)} Kohaerenzzeit(en) evaluiert.")


if __name__ == "__main__":
    evaluate_fixed_actions(
        case_study_id=CASE_STUDY_ID,
        fixed_actions=FIXED_ACTIONS,
        n_episodes=N_EPISODES,
        coherence_times=COHERENCE_TIMES if COHERENCE_TIMES else None,
    )
