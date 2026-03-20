"""
evaluate_replace_threshold.py
─────────────────────────────
Findet für jede Case Study und jede Coherence Time den Fidelity-Schwellwert
(f_mem), ab dem REPLACE (= KEEP BEST) die wahrscheinlichste Aktion ist.

Ansatz: Statt echte Episoden zu simulieren werden synthetische Observations
direkt ans Policy-Netz übergeben. f_mem wird von 0.5 → 1.0 gesweept.

Swept wird pro (case_study_id, coherence_time, t_req):
  obs = [f_mem, 0.0, t_req, λ1, λ2, λ3]
  mit t_req ∈ T_REQ_VALUES  (0.0, 0.01, 0.05, 0.12)
  und λ1, λ2, λ3 aus dem ConstantsTuple der Case Study

Ausgabe je Case Study:
  results/case_study_{id}/replace_threshold.csv
  Spalten: coherence_time, time_since_last_req, replace_threshold_f_mem
"""

import os
import csv
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from ppo.case_studies import CASE_STUDIES, CaseStudy
from ppo.custom_env import TrainingEnv
from ppo.trainer import coherence_time_folder, case_study_root

# ─── Konfiguration ────────────────────────────────────────────────────────────

# Case Studies, die evaluiert werden sollen (None = alle)
CASE_STUDY_IDS: list[int] | None = [2]

# t_req-Werte, für die der Sweep gemacht wird (bleibt request_waiting = 0)
T_REQ_VALUES: list[float] = [0.0, 0.01, 0.05, 0.12]

# f_mem-Sweep-Auflösung
F_MEM_STEPS: int = 2000

# Welches Modell evaluiert wird
MODEL_FILE = "best_model.zip"

# ─── Kern-Funktion ────────────────────────────────────────────────────────────

def find_replace_threshold(
    policy,
    lambdas: tuple[float, float, float],
    t_req: float,
    f_mem_steps: int = F_MEM_STEPS,
) -> float | None:
    """
    Sweept f_mem von 0.5 → 1.0 und gibt zurück:
      - replace_threshold: erstes f_mem, bei dem REPLACE die höchste prob hat
        (None falls REPLACE nie die höchste Prob hat)
      - p_replace_at_threshold: p(REPLACE) an diesem Punkt
      - f_mem_values, p_replace_values: vollständige Sweepkurven (für Plots)
    """
    l1, l2, l3 = lambdas
    f_mem_values    = np.linspace(0.5, 1.0, f_mem_steps)
    p_replace_vals  = []
    threshold_f_mem = None
    threshold_p     = None

    for f_mem in f_mem_values:
        obs = np.array([f_mem, 0.0, t_req, l1, l2, l3], dtype=np.float32)
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist  = policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.cpu().numpy()[0]

        p_replace = float(probs[0])
        p_replace_vals.append(p_replace)

        # Schwellwert: erstes f_mem, bei dem REPLACE die höchste prob hat
        if threshold_f_mem is None and int(np.argmax(probs)) == 0:
            threshold_f_mem = float(f_mem)
            threshold_p     = p_replace

    return threshold_f_mem


def save_threshold_csv(case_study_id: int, rows: list[dict]) -> None:
    root = case_study_root(case_study_id)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "replace_threshold.csv")

    fieldnames = ["coherence_time", "time_since_last_req", "replace_threshold_f_mem"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  💾  Gespeichert: {path}")


# ─── Haupt-Evaluierung ────────────────────────────────────────────────────────

def evaluate_replace_threshold(case_study_id: int) -> None:
    case_study = CASE_STUDIES[case_study_id]
    lambdas = case_study.constants.lambdas  # (λ1, λ2, λ3)

    print(f"\n{'═' * 58}")
    print(f"  REPLACE-Threshold  |  Case Study {case_study_id}  |  λ={lambdas}")
    print(f"{'═' * 58}")

    all_rows: list[dict] = []

    for coherence_time in case_study.coherence_times:
        ct_folder = coherence_time_folder(case_study_id, coherence_time)
        ct_str    = CaseStudy.coherence_time_str(coherence_time)
        model_path = os.path.join(ct_folder, MODEL_FILE)
        print(f"\n  T_coh = {coherence_time} ({ct_str})", end="")

        if not os.path.exists(model_path):
            print(f"  ⚠️  {MODEL_FILE} nicht gefunden, übersprungen.")
            continue

        # Lade Modell + Policy (kein volles Env-Step nötig)
        constants = case_study.make_constants(coherence_time)
        env       = TrainingEnv(constants)
        try:
            model = PPO.load(model_path, env=env, device="cpu")
        except (ValueError, Exception) as e:
            print(f"  ❌  Laden fehlgeschlagen ({e}), übersprungen.")
            continue

        policy = model.policy
        policy.eval()
        print()

        for t_req in T_REQ_VALUES:
            th = find_replace_threshold(policy, lambdas, t_req)

            if th is not None:
                print(f"    t_req={t_req:5.2f}  →  f_th = {th:.4f}")
            else:
                print(f"    t_req={t_req:5.2f}  →  REPLACE nie dominant")

            all_rows.append({
                "coherence_time":          coherence_time,
                "time_since_last_req":     t_req,
                "replace_threshold_f_mem": th,
            })

    save_threshold_csv(case_study_id, all_rows)
    print(f"\n  ✅  Case Study {case_study_id} abgeschlossen.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ids = CASE_STUDY_IDS if CASE_STUDY_IDS is not None else list(CASE_STUDIES.keys())
    for cs_id in ids:
        evaluate_replace_threshold(cs_id)
