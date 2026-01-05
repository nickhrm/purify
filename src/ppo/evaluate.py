import os

import matplotlib.pyplot as plt
import numpy as np

from ppo.custom_env import TrainingEnv
from ppo.policy_wrapper import FixedActionAgent, PPOAgent, RandomAgent
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy


def run_parameter_sweep():
    decoherence_times = [10e-4, 5e-3, 10e-3, 5e-2, 10e-2]
    N_EPISODES = 50

    # Dictionary initialisieren wir jetzt dynamisch
    results = {}

    print(f"Starte Sweep über {len(decoherence_times)} Zeitwerte...")

    for t_coh in decoherence_times:
        print(f"Evaluating for T_coh = {t_coh}...", end="\r")

        current_constants = ConstantsTuple(
            decoherence_time=t_coh,
            strategy=Action.TRAINING_MODE,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            waiting_time_sensitivity=1,
            pumping_probability=1.0,
        )

        env = TrainingEnv(current_constants)

        # Hier definierst du deine Strategien an EINEM Ort
        policies = [
            PPOAgent("ppo_quantum_agent", env),
            FixedActionAgent(Action.REPLACE),
            FixedActionAgent(Action.PMD),
            # FixedActionAgent(Action.PROT_1),
            # FixedActionAgent(Action.PROT_2),
            # FixedActionAgent(Action.PROT_3),
            # RandomAgent(env.action_space)
        ]

        for policy in policies:

            if policy.name not in results:
                results[policy.name] = []

            print(f"Testing {policy.name}")
            total_reward = 0.0
            for _ in range(N_EPISODES):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = policy.predict(obs)
                    print(f"Taking action {Action(action).name}")
                    obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    done = terminated or truncated

            results[policy.name].append(total_reward / N_EPISODES)

    print("\nSweep beendet.")
    return decoherence_times, results

def plot_sweep(x, y_data):
    plt.figure(figsize=(10, 6))
    for name, values in y_data.items():
        plt.plot(x, values, label=name, marker='o', markersize=4)

    plt.xscale('log')
    plt.xlabel('Decoherence Time ($T_{coh}$)')
    plt.ylabel('Average Teleportation Fidelity (Reward)')
    plt.title('Agent Performance vs. Decoherence Time')
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    # Pfad ohne führenden Slash für lokale Speicherung
    plt.savefig("sweep_results.png")
    print("Graph gespeichert als 'sweep_results.png'")

if __name__ == "__main__":
    times, res = run_parameter_sweep()
    plot_sweep(times, res)
