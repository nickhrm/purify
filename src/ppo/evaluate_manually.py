import torch
import numpy as np
import pandas as pd
import os
from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import SB3Agent
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy

def analyze_fidelity_dependency(
    case_study_id,
    coherence_time,
    lambdas,
    output_file="fidelity_analysis.csv"
):
    # 1. Dynamischer Model-Pfad basierend auf Parameter
    # Wandelt z.B. 0.05 in "0_05" um
    ct_str = str(coherence_time).replace('.', '_')
    model_path = f"results/case_study_{case_study_id}/T{ct_str}/best_model.zip"

    # 2. Setup Environment & Agent
    temp_constants = ConstantsTuple(
        coherence_time=coherence_time,
        lambda_strategy=LambdaSrategy.USE_CONSTANTS,
        waiting_time_sensitivity=1,
        pumping_probability=1.0,
        lambdas=lambdas, # <-- Nutzt die übergebenen Lambdas
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

    # 3. Fidelity-Bereich definieren
    fidelities = [0.6, 0.7, 0.8, 0.9, 1.0]
    waiting_times = [0, 0.01, 0.05, 0.12]

    action_names = [a.name for a in temp_constants.actions]

    results = []

    # Äußere Schleife
    for waiting_time in waiting_times:
        # Innere Schleife
        for fid in fidelities:
            # 4. KÜNSTLICHE OBSERVATION (Dynamisch mit temp_constants.lambdas)
            obs = np.array([
                fid, 
                0.0, 
                waiting_time, 
                temp_constants.lambdas[0], 
                temp_constants.lambdas[1], 
                temp_constants.lambdas[2]
            ], dtype=np.float64)

            # 5. PROBABILITIES EXTRAHIEREN
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
                "model_path": model_path, 
                "waiting_time": waiting_time
            }
            for i, p in enumerate(probs):
                row[f"prob_{action_names[i]}"] = p

            results.append(row)

    # 6. APPEND-LOGIK
    df = pd.DataFrame(results)
    
    # Prüfen, ob Datei existiert
    file_exists = os.path.isfile(output_file)
    
    # mode='a' hängt an, header=False wenn die Datei schon existiert
    df.to_csv(output_file, mode='a', index=False, header=not file_exists)
    
    print(f"Daten für T_coh={coherence_time} (Lambdas={lambdas}) an {output_file} angehängt. (Insgesamt {len(df)} Zeilen geschrieben.)")

if __name__ == "__main__":
    # Beispielaufrufe: So kannst du ganz einfach alle durchlaufen lassen!
    analyze_fidelity_dependency(case_study_id=3, coherence_time=0.003, lambdas=(0.0, 0.0, 0.3))
    
    # Für die anderen Case Studies müsstest du nur das hier aufrufen:
    # analyze_fidelity_dependency(case_study_id=1, coherence_time=0.05, lambdas=(0.3, 0.0, 0.0))
    # analyze_fidelity_dependency(case_study_id=2, coherence_time=0.05, lambdas=(0.0, 0.3, 0.0))