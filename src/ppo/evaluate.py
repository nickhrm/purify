import csv
import os
import resource
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.utils.actor_manager import CallResult

# Deine Imports
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent, FixedActionAgent
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy


def save_average_f(times, results, filename="sweep_results.csv"):
    """
    Speichert die Ergebnisse im Format: coherence_time, fidelity, model.
    Hängt Daten an, wenn die Datei schon existiert (Append-Modus).
    """
    file_exists = os.path.isfile(filename)

    # mode='a' für Append
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Header nur schreiben, wenn Datei neu ist
        if not file_exists:
            writer.writerow(["coherence_time", "fidelity", "model"])

        # Iteriere über alle Modelle in diesem Batch
        for model_name, fidelities in results.items():
            # Sicherstellen, dass wir Daten haben
            if not fidelities:
                continue

            # Wir nutzen zip, um Zeit und Fidelity zu paaren
            # times muss hier exakt so lang sein wie fidelities
            for t, fid in zip(times, fidelities):
                writer.writerow([t, fid, model_name])

    print(f"Daten an {filename} angehängt.")


def save_action_distribution(action_counts, coherence_time, model, filename="action_prob.csv"):
    file_exists = os.path.isfile(filename)

    total_actions = sum(action_counts.values())
    
    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)

        # Header only if file is new
        if not file_exists:
            writer.writerow(["model", "coherence_time", "action", "count", "percentage"])
        
        for action_name, count in action_counts.items():
            percentage = (count / total_actions) * 100 if total_actions > 0 else 0
            writer.writerow([model, coherence_time, action_name, count, percentage])



def run_parameter_sweep():
    # Dein Test-Szenario
    test_config = {
        "0_001" : [0.001],
        "0_002" : [0.002],
        "0_003" : [0.003],
        "0_004" : [0.004],
        "0_005" : [0.005],
        "0_006" : [0.006],
        "0_007" : [0.007],
        "0_008" : [0.008],
        "0_009" : [0.009],
        "0_01" : [0.010],
        "0_02" : [0.020],
        "0_03" : [0.030],
        "0_04" : [0.040],
        "0_05": [0.05],
        "0_06": [0.06],
        "0_07": [0.07],
        "0_08": [0.08],
        "0_09": [0.09],
        #  "0_1": [0.1],



    }

    N_EPISODES = 600
    CSV_FILENAME = "sweep_results.csv"

    # Falls die Datei vom vorherigen Run noch da ist und stört,
    # könnte man sie hier löschen. Wenn du sammeln willst, lass es so.
    # if os.path.exists(CSV_FILENAME):
    #     os.remove(CSV_FILENAME)

    print(f"Starte Sweep über {len(test_config)} Modell-Konfigurationen...")

    # 1. ÄUSSERE SCHLEIFE: Iteriere über die Modelle (Keys)
    for model_folder, times_list in test_config.items():
        print(f"\n--- Teste Modell aus Ordner: {model_folder} ---")

        # Pfad dynamisch zusammenbauen
        model_path = f"best_models_5gps_03_00_00/{model_folder}/best_model.zip"
        # Temporärer Speicher für DIESEN Batch (nur dieses Modell + Baselines für diese Zeiten)
        batch_results = {}

        # 2. INNERE SCHLEIFE: Iteriere über die Zeiten für dieses Modell
        for t_coh in times_list:
            print(f"  Evaluating T_coh = {t_coh}...", end="\r")

            # Environment Setup
            current_constants = ConstantsTuple(
                coherence_time=t_coh,
                lambda_strategy=LambdaSrategy.USE_CONSTANTS,
                waiting_time_sensitivity=1,
                pumping_probability=1.0,
                lambdas=(0.3, 0.0, 0.0),
            )
            env = TrainingEnv(current_constants)

            policies = [
                SB3Agent(model_path, env),
                # FixedActionAgent(Action.REPLACE),
                # FixedActionAgent(Action.PROT_1),
                # FixedActionAgent(Action.PROT_2),
                # FixedActionAgent(Action.PROT_3),
                # FixedActionAgent(Action.PMD)
            ]


            # Evaluation Loop
            for policy in policies:
                if policy.name not in batch_results:
                    batch_results[policy.name] = []

                total_reward = 0.0
                action_counts = Counter()
                
                for _ in range(N_EPISODES):
                    obs, _ = env.reset()
                    done = False
                    while not done:
                        action = policy.predict(obs)
                        action_name = Action(action).name
                        action_counts[action_name] += 1
                        # save_actions(Action(action).name, current_constants.coherence_time, policy.name)
                        obs, reward, terminated, truncated, _ = env.step(action)
                        total_reward += reward
                        done = terminated or truncated
                
                # Save aggregated action stats for this policy & coherence_time
                save_action_distribution(action_counts, current_constants.coherence_time, policy.name)

                avg_reward = total_reward / N_EPISODES
                batch_results[policy.name].append(avg_reward)

        # 3. SPEICHERN: Nach jedem Modell-Block schreiben wir in die CSV
        # times_list sind die X-Werte, batch_results die Y-Werte für diesen Block
        save_average_f(times_list, batch_results, CSV_FILENAME)

    print("\nSweep beendet.")


if __name__ == "__main__":
    run_parameter_sweep()

