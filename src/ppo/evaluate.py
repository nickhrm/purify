import csv
import os
from collections import Counter

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent, FixedActionAgent
from ppo.trainer import case_study_root, coherence_time_folder


def save_evaluation_results(case_study_id: int, rows: list[dict]) -> None:
    """
    Schreibt / hängt Evaluierungsergebnisse an
    results/case_study_{id}/evaluation.csv an.
    """
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    filename = os.path.join(root, "evaluation.csv")
    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        fieldnames = ["case_study_id", "coherence_time", "model", "avg_reward"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Ergebnisse gespeichert: {filename}")


def save_action_distribution(
    action_counts: Counter,
    coherence_time: float,
    model_name: str,
    case_study_id: int,
    filename: str = "action_prob.csv",
) -> None:
    file_exists = os.path.isfile(filename)
    total_actions = sum(action_counts.values())

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["case_study_id", "model", "coherence_time", "action", "count", "percentage"])
        for action_name, count in action_counts.items():
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            writer.writerow([case_study_id, model_name, coherence_time, action_name, count, percentage])


# ─── Haupt-Evaluierungs-Funktion ──────────────────────────────────────────────

def evaluate_case_study(case_study_id: int, n_episodes: int = 1200) -> None:
    """
    Evaluiert alle Kohärenzzeiten einer Case Study.
    Pro coherence_time werden sowohl best_model.zip als auch end_model.zip getestet
    (falls vorhanden).

    Ergebnisse landen in:
      results/case_study_{id}/evaluation.csv
    """
    case_study = CASE_STUDIES[case_study_id]
    print(f"\n{'═' * 55}")
    print(f"  Evaluiere Case Study {case_study_id}")
    print(f"  Kohärenzzeiten: {case_study.coherence_times}")
    print(f"  Episoden pro Modell: {n_episodes}")
    print(f"{'═' * 55}\n")

    all_rows: list[dict] = []

    for coherence_time in case_study.coherence_times:
        constants = case_study.make_constants(coherence_time)
        ct_folder = coherence_time_folder(case_study_id, coherence_time)
        ct_str = CaseStudy.coherence_time_str(coherence_time)
        print(f"\n--- T_coh = {coherence_time} ({ct_str}) ---")

        # Kandidaten: best_model und end_model
        model_candidates = {
            "best_model": os.path.join(ct_folder, "best_model.zip"),
            "end_model": os.path.join(ct_folder, "end_model.zip"),
        }

        env = TrainingEnv(constants)

        for model_label, model_path in model_candidates.items():
            if not os.path.exists(model_path):
                print(f"  ⚠️  {model_label} nicht gefunden, übersprungen: {model_path}")
                continue

            print(f"  Evaluiere {model_label}...", end="", flush=True)
            agent = SB3Agent(model_path, env)
            action_counts = Counter()
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

            avg_reward = total_reward / n_episodes
            print(f"  avg_reward = {avg_reward:.4f}")

            save_action_distribution(
                action_counts,
                coherence_time,
                model_label,
                case_study_id,
            )

            all_rows.append({
                "case_study_id": case_study_id,
                "coherence_time": coherence_time,
                "model": model_label,
                "avg_reward": avg_reward,
            })

    save_evaluation_results(case_study_id, all_rows)
    print(f"\n✅ Evaluierung Case Study {case_study_id} abgeschlossen.")


if __name__ == "__main__":
    evaluate_case_study(case_study_id=4, n_episodes=2500)
