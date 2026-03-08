import torch
import numpy as np
import pandas as pd
import os
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy

def analyze_fidelity_dependency(
    coherence_time=0.05, output_file="fidelity_analysis.csv"
):
    model_path = f"results/case_study_1/T0_01/best_model.zip"

    # 1. Setup Environment & Agent
    temp_constants = ConstantsTuple(
        coherence_time=coherence_time,
        lambda_strategy=LambdaSrategy.USE_CONSTANTS,
        waiting_time_sensitivity=1,
        pumping_probability=1.0,
        lambdas=(0.0, 0.3, 0.0),
        actions=(
            Action.REPLACE,
            Action.PROT_1,
            Action.PROT_2,
            Action.PROT_3,
        ),
        min_fidelity=0.7,
        max_fidelity=0.7,
    )
    env = TrainingEnv(temp_constants)

    if not os.path.exists(model_path):
        print(f"Fehler: Modell unter {model_path} nicht gefunden.")
        return

    agent = SB3Agent(model_path, env)
    model = agent.model
    policy = model.policy

    # 2. Fidelity-Bereich definieren
    fidelities = [ 0.6, 0.7, 0.8, 0.9, 1.0]
    waiting_times = [0, 0.01, 0.05, 0.12]

    action_names = [a.name for a in temp_constants.actions]

    results = []

    for waiting_time in waiting_times:
        for fid in fidelities:
            # 3. KÜNSTLICHE OBSERVATION
            # Dein definierter State-Vektor
            obs = np.array([fid, 0.0, waiting_time, 0.3, 0.0, 0.0], dtype=np.float64)

            # 4. PROBABILITIES EXTRAHIEREN
            obs_tensor = torch.as_tensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                distribution = policy.get_distribution(obs_tensor)
                probs = distribution.distribution.probs.cpu().numpy()[0]
                best_action_idx = np.argmax(probs)
                best_action_name = action_names[best_action_idx]

            row = {
                "fidelity": fid,
                "best_action": best_action_name,
                "coherence_time": coherence_time,
                "model_path": model_path, # Hilfreich beim Appenden, um Daten zuzuordnen
                "waiting_time": waiting_time
            }
            for i, p in enumerate(probs):
                row[f"prob_{action_names[i]}"] = p

            results.append(row)

        # 5. APPEND-LOGIK
        df = pd.DataFrame(results)
        
        # Prüfen, ob Datei existiert
        file_exists = os.path.isfile(output_file)
        
        # mode='a' hängt an, header=False wenn die Datei schon existiert
        df.to_csv(output_file, mode='a', index=False, header=not file_exists)
        
        print(f"Daten für T_coh={coherence_time} an {output_file} angehängt.")

if __name__ == "__main__":
    # Du kannst hier jetzt z.B. eine Liste von Zeiten durchlaufen lassen
        analyze_fidelity_dependency()