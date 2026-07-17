import argparse
import csv
import os
from collections import Counter

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent, SymmetrizedSB3Agent
from ppo.trainer import case_study_root, coherence_time_folder
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaStrategy


def save_evaluation_results(
    case_study_id: int, rows: list[dict], csv_name: str = "evaluation.csv"
) -> None:
    """
    Schreibt / hängt Evaluierungsergebnisse an
    results/case_study_{id}/{csv_name} an.
    """
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    filename = os.path.join(root, csv_name)
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
    csv_name: str = "action_prob.csv",
) -> None:
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    filename = os.path.join(root, csv_name)
    file_exists = os.path.isfile(filename)
    total_actions = sum(action_counts.values())

    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "case_study_id",
                    "model",
                    "coherence_time",
                    "action",
                    "count",
                    "percentage",
                ]
            )
        for action_name, count in action_counts.items():
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            writer.writerow(
                [
                    case_study_id,
                    model_name,
                    coherence_time,
                    action_name,
                    count,
                    percentage,
                ]
            )


# ─── Haupt-Evaluierungs-Funktion ──────────────────────────────────────────────


def run_episodes(
    env: TrainingEnv,
    agent: SB3Agent,
    constants: ConstantsTuple,
    n_episodes: int,
    seed: int | None = None,
) -> tuple[float, Counter]:
    """Führt n_episodes Episoden aus und gibt (avg_reward, action_counts) zurück.
    Mit gesetztem `seed` ist der Umgebungs-Zufallsstrom reproduzierbar."""
    action_counts = Counter()
    total_reward = 0.0

    env.reset(seed=seed)
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

    return total_reward / n_episodes, action_counts


def evaluate_case_study(
    case_study_id: int, n_episodes: int = 1200, seed: int | None = None
) -> None:
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
            agent = SB3Agent(model_path, env, deterministic=case_study.deterministic)
            avg_reward, action_counts = run_episodes(env, agent, constants, n_episodes, seed=seed)
            print(f"  avg_reward = {avg_reward:.4f}")

            save_action_distribution(
                action_counts,
                coherence_time,
                model_label,
                case_study_id,
            )

            all_rows.append(
                {
                    "case_study_id": case_study_id,
                    "coherence_time": coherence_time,
                    "model": model_label,
                    "avg_reward": avg_reward,
                }
            )

    save_evaluation_results(case_study_id, all_rows)
    print(f"\n✅ Evaluierung Case Study {case_study_id} abgeschlossen.")


# ─── Generalisierungs-Test: fixe Trainings-Lambdas → zufällige Fehlerverteilung ─


def evaluate_generalization_to_random_lambdas(
    case_study_id: int,
    n_episodes: int = 1200,
    coherence_times: list[float] | None = None,
    seed: int | None = None,
    symmetrized: bool = False,
) -> None:
    """
    Testet, wie gut Modelle generalisieren, die nur auf einer festen
    Fehlerverteilung trainiert wurden (z.B. λ=(0.3, 0.0, 0.0)).

    Die Modelle der Case Study werden dafür in einer Umgebung mit
    LambdaStrategy.RANDOM evaluiert: Die initiale Fidelity bleibt dieselbe
    wie im Training (1 - Σλ), aber die Fehlermasse wird pro Entanglement
    zufällig auf λ₁, λ₂, λ₃ verteilt. Das entspricht der Umgebung der
    RANDOM-Case-Studies (z.B. 4 und 10), sodass die Ergebnisse direkt mit
    deren evaluation.csv vergleichbar sind.

    Mit `symmetrized=True` wird die Policy per Permutations-Symmetrie
    äquivariant gemacht (siehe SymmetrizedSB3Agent): λ der Observation werden
    absteigend sortiert (kanonische Trainings-Form), die gewählte PROT-Aktion
    wird zurückpermutiert. Kein Retraining nötig.

    Ergebnisse landen in:
      results/case_study_{id}/evaluation_random_lambdas[_sym].csv
      action_prob_random_lambdas[_sym].csv
    """
    case_study = CASE_STUDIES[case_study_id]
    if coherence_times is None:
        coherence_times = case_study.coherence_times

    suffix = "_sym" if symmetrized else ""

    if case_study.constants.lambda_strategy == LambdaStrategy.RANDOM:
        print(
            "⚠️  Diese Case Study wurde bereits auf RANDOM trainiert – "
            "der Generalisierungs-Test ist nur für fixe Lambdas sinnvoll."
        )

    training_fidelity = 1.0 - sum(case_study.constants.lambdas)

    if Action.PMD in case_study.constants.actions:
        print(
            "⚠️  Achtung: PMD ist nur für λ₂ = λ₃ = 0 definiert. Wählt das "
            "Modell PMD bei zufälligen Lambdas, bricht die Evaluation ab."
        )

    print(f"\n{'═' * 55}")
    print(f"  Generalisierungs-Test Case Study {case_study_id}")
    print(f"  Training: λ = {case_study.constants.lambdas} (fix)")
    print(f"  Evaluation: F = {training_fidelity}, Fehler zufällig verteilt")
    print(f"  Symmetrisierte Policy: {'ja' if symmetrized else 'nein'}")
    print(f"  Kohärenzzeiten: {coherence_times}")
    print(f"  Episoden pro Modell: {n_episodes}")
    print(f"{'═' * 55}\n")

    all_rows: list[dict] = []

    for coherence_time in coherence_times:
        eval_constants = case_study.make_constants(coherence_time)._replace(
            lambda_strategy=LambdaStrategy.RANDOM,
            min_fidelity=training_fidelity,
            max_fidelity=training_fidelity,
        )
        ct_folder = coherence_time_folder(case_study_id, coherence_time)
        print(f"\n--- T_coh = {coherence_time} ---")

        model_candidates = {
            "best_model": os.path.join(ct_folder, "best_model.zip"),
            "end_model": os.path.join(ct_folder, "end_model.zip"),
        }

        env = TrainingEnv(eval_constants)

        for model_label, model_path in model_candidates.items():
            if not os.path.exists(model_path):
                print(f"  ⚠️  {model_label} nicht gefunden, übersprungen: {model_path}")
                continue

            print(f"  Evaluiere {model_label}...", end="", flush=True)
            if symmetrized:
                agent = SymmetrizedSB3Agent(
                    model_path,
                    env,
                    actions=eval_constants.actions,
                    deterministic=case_study.deterministic,
                )
            else:
                agent = SB3Agent(model_path, env, deterministic=case_study.deterministic)
            avg_reward, action_counts = run_episodes(
                env, agent, eval_constants, n_episodes, seed=seed
            )
            print(f"  avg_reward = {avg_reward:.4f}")

            save_action_distribution(
                action_counts,
                coherence_time,
                model_label,
                case_study_id,
                csv_name=f"action_prob_random_lambdas{suffix}.csv",
            )

            all_rows.append(
                {
                    "case_study_id": case_study_id,
                    "coherence_time": coherence_time,
                    "model": model_label,
                    "avg_reward": avg_reward,
                }
            )

    save_evaluation_results(
        case_study_id, all_rows, csv_name=f"evaluation_random_lambdas{suffix}.csv"
    )
    print(f"\n✅ Generalisierungs-Test Case Study {case_study_id} abgeschlossen.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluiert die trainierten Modelle einer Case Study.",
        epilog=(
            "Beispiele: "
            "'evaluate --case-study 1' (normale Evaluation) | "
            "'evaluate --case-study 1 --random-lambdas' (Generalisierungs-Test: "
            "Modelle mit fixen Trainings-Lambdas unter zufällig verteiltem Fehler)"
        ),
    )
    parser.add_argument(
        "--case-study",
        type=int,
        required=True,
        choices=sorted(CASE_STUDIES),
        help="ID der Case Study (siehe ppo/case_studies.py)",
    )
    parser.add_argument(
        "--episodes", type=int, default=1200, help="Episoden pro Modell (Default: 1200)"
    )
    parser.add_argument(
        "--coherence-times",
        type=float,
        nargs="+",
        default=None,
        metavar="T",
        help="Nur diese Kohärenzzeiten evaluieren; Default: alle der Case Study "
        "(nur für --random-lambdas)",
    )
    parser.add_argument(
        "--random-lambdas",
        action="store_true",
        help="Generalisierungs-Test: in Umgebung mit zufällig verteiltem "
        "Fehler evaluieren statt in der Trainings-Umgebung",
    )
    parser.add_argument(
        "--symmetrized",
        action="store_true",
        help="Nur mit --random-lambdas: Policy per Permutations-Symmetrie "
        "äquivariant machen (λ kanonisch sortieren, PROT-Aktion zurückpermutieren)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed für reproduzierbare Evaluation (Default: zufällig)",
    )
    args = parser.parse_args()

    if args.symmetrized and not args.random_lambdas:
        parser.error("--symmetrized ist nur zusammen mit --random-lambdas sinnvoll")

    if args.random_lambdas:
        evaluate_generalization_to_random_lambdas(
            case_study_id=args.case_study,
            n_episodes=args.episodes,
            coherence_times=args.coherence_times,
            seed=args.seed,
            symmetrized=args.symmetrized,
        )
    else:
        evaluate_case_study(case_study_id=args.case_study, n_episodes=args.episodes, seed=args.seed)


if __name__ == "__main__":
    main()
