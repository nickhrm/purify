import ray
from ray.rllib.algorithms import Algorithm
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy

def run_rllib_native_sweep():
    # Ray starten
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "excludes": [
                    # Exclude large .pack files in .git to avoid upload errors
                    "**/.git/objects/pack/*.pack",
                ]
            }
        )

    base_path = "/home/nick/Documents/purify/ppo_results_rllib/checkpoints/0_001/best_model"
    algo = Algorithm.from_checkpoint(base_path)

    # Die Zeiten, die wir testen wollen
    test_times = [0.001, 0.002, 0.005, 0.010]

    print(f"Starte Native RLlib Sweep für Modell: {base_path}")

    for t_coh in test_times:
        # 1. Neue Konstanten definieren
        new_constants = ConstantsTuple(
            coherence_time=t_coh,  # <--- HIER ändern wir den Wert
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
        )

        # 2. MAGIE: Wir senden die neuen Konstanten an ALLE Evaluation-Worker
        # foreach_env führt eine Funktion auf jedem Environment in jedem Worker aus.
        algo.evaluation_workers.foreach_env(
            lambda env: env.set_constants(new_constants)
        )

        # 3. Jetzt evaluieren (RLlib nutzt nun die neuen Konstanten)
        # Wir können auch temporär die Anzahl der Episoden ändern
        results = algo.evaluate()
        
        # 4. Daten holen
        avg_ret = results["env_runners"]["episode_return_mean"]
        print(f"Coherence Time: {t_coh} -> Reward: {avg_ret:.4f}")

    ray.shutdown()

if __name__ == "__main__":
    run_rllib_native_sweep()