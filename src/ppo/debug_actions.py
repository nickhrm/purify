"""
debug_actions.py
────────────────
Läuft N Episoden mit dem trainierten 4-GPS Modell (Case Study 2, t_c=0.05)
und loggt für jeden Entscheidungsmoment:
  - Die komplette Observation (f_mem, request_is_waiting, t_req, l1, l2, l3)
  - Die gewählte Action
  - Die Action-Probabilities des Modells
  - Die berechnete Teleportations-Fidelity (nur wenn in dieser Episode)

Ausgabe: debug_action_log.csv
"""

import os
import torch
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy
from purify.utils.purification_util import Purification

# ── Konfiguration ──────────────────────────────────────────────────────────────
COHERENCE_TIME = 0.05
CASE_STUDY_ID  = 4
N_EPISODES     = 5          # Wenige Episoden, dafür vollständig geloggt
MODEL_PATH     = f"results/case_study_{CASE_STUDY_ID}/T0_05/best_model.zip"
OUTPUT_FILE    = "debug_action_log.csv"
# ──────────────────────────────────────────────────────────────────────────────


def run_debug():
    constants = ConstantsTuple(
        coherence_time=COHERENCE_TIME,
        pumping_probability=1,
        waiting_time_sensitivity=1,
        lambda_strategy=LambdaSrategy.USE_CONSTANTS,
        lambdas=(0.0, 0.0, 0.3),
        actions=(Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3),
        min_fidelity=0.7,
        max_fidelity=0.7,
    )
    env = TrainingEnv(constants)

    if not os.path.exists(MODEL_PATH):
        print(f"❌  Modell nicht gefunden: {MODEL_PATH}")
        return

    model = PPO.load(MODEL_PATH, env=env, device="cpu")
    policy = model.policy
    action_names = [a.name for a in constants.actions]

    records = []

    for episode in range(N_EPISODES):
        obs, _ = env.reset()
        done = False
        step_in_episode = 0
        episode_reward = None  # gesetzt sobald Teleportation stattfindet

        while not done:
            f_mem, req_waiting, t_req, l1, l2, l3 = obs

            # ── Action + Probabilities vom Modell ──────────────────────────
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).float()
            with torch.no_grad():
                dist  = policy.get_distribution(obs_tensor)
                probs = dist.distribution.probs.cpu().numpy()[0]

            action_idx, _ = model.predict(obs, deterministic=True)
            action_idx    = int(action_idx)
            action_name   = action_names[action_idx]

            # ── PROT_3 spezifische Zusatzinfos berechnen ───────────────────
            # (nur wenn good_memory vorhanden, was zu diesem Zeitpunkt immer
            #  True ist — der Agent wird nur bei needs_agent_decision() aufgerufen)
            node = env.node
            prot3_gain      = None
            prot3_prob      = None
            prot1_gain      = None
            prot1_prob      = None
            keep_best_delta = None  # wie viel besser/schlechter ist das neue Entanglement

            if env.last_generated_entanglement is not None and node.good_memory is not None:
                e_good  = node.good_memory
                e_new   = env.last_generated_entanglement
                f_good  = e_good.get_current_fidelity()
                f_new   = e_new.get_current_fidelity()
                keep_best_delta = f_new - f_good  # positiv → neues ist besser

                # PROT_3
                try:
                    prot3_f_after = Purification.jump_function_from_action(e_good, e_new, Action.PROT_3)
                    prot3_p       = Purification.success_probability_from_action(e_good, e_new, Action.PROT_3)
                    prot3_gain    = prot3_f_after - f_good
                    prot3_prob    = prot3_p
                except Exception:
                    pass

                # PROT_1
                try:
                    prot1_f_after = Purification.jump_function_from_action(e_good, e_new, Action.PROT_1)
                    prot1_p       = Purification.success_probability_from_action(e_good, e_new, Action.PROT_1)
                    prot1_gain    = prot1_f_after - f_good
                    prot1_prob    = prot1_p
                except Exception:
                    pass

            # ── Env-Step ───────────────────────────────────────────────────
            obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            if reward > 0:
                episode_reward = reward

            records.append({
                "episode":          episode,
                "step":             step_in_episode,
                # Observation
                "f_mem":            round(float(f_mem), 4),
                "request_waiting":  int(req_waiting),
                "t_req":            round(float(t_req), 4),
                "l1":               round(float(l1), 4),
                "l2":               round(float(l2), 4),
                "l3":               round(float(l3), 4),
                # Entscheidung
                "action":           action_name,
                # Probabilities
                "p_REPLACE":        round(float(probs[0]), 4),
                "p_PROT_1":         round(float(probs[1]), 4),
                "p_PROT_2":         round(float(probs[2]), 4),
                "p_PROT_3":         round(float(probs[3]), 4),
                # Purification-Kalkulation
                "keep_best_delta":  round(float(keep_best_delta), 4) if keep_best_delta is not None else None,
                "prot1_gain":       round(float(prot1_gain), 4)      if prot1_gain      is not None else None,
                "prot1_prob":       round(float(prot1_prob), 4)       if prot1_prob      is not None else None,
                "prot3_gain":       round(float(prot3_gain), 4)       if prot3_gain      is not None else None,
                "prot3_prob":       round(float(prot3_prob), 4)       if prot3_prob      is not None else None,
                # Ergebnis
                "teleport_fidelity": round(float(episode_reward), 4) if (done and episode_reward is not None) else None,
            })
            step_in_episode += 1

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅  Log gespeichert: {OUTPUT_FILE}  ({len(df)} Zeilen, {N_EPISODES} Episoden)")

    # ── Kurze Zusammenfassung ──────────────────────────────────────────────
    print("\n── Action-Verteilung ──────────────────────────────────")
    print(df["action"].value_counts(normalize=True).map("{:.1%}".format).to_string())

    print("\n── Wann wird PROT_3 gewählt? ─────────────────────────")
    prot3_df = df[df["action"] == "PROT_3"]
    replace_df = df[df["action"] == "REPLACE"]

    print(f"\n  PROT_3  (n={len(prot3_df)}):")
    print(f"    f_mem mean:          {prot3_df['f_mem'].mean():.4f}")
    print(f"    t_req mean:          {prot3_df['t_req'].mean():.4f}")
    print(f"    request_waiting:     {prot3_df['request_waiting'].mean():.2%} der Fälle")
    print(f"    prot3_gain mean:     {prot3_df['prot3_gain'].mean():.4f}")
    print(f"    keep_best_delta mean:{prot3_df['keep_best_delta'].mean():.4f}")

    print(f"\n  REPLACE (n={len(replace_df)}):")
    print(f"    f_mem mean:          {replace_df['f_mem'].mean():.4f}")
    print(f"    t_req mean:          {replace_df['t_req'].mean():.4f}")
    print(f"    request_waiting:     {replace_df['request_waiting'].mean():.2%} der Fälle")
    print(f"    prot3_gain mean:     {replace_df['prot3_gain'].mean():.4f}")
    print(f"    keep_best_delta mean:{replace_df['keep_best_delta'].mean():.4f}")

    print("\n── Teleportations-Fidelity nach letzter Action ────────")
    last_steps = df[df["teleport_fidelity"].notna()]
    print(last_steps.groupby("action")["teleport_fidelity"].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    run_debug()
