import atexit  # Neu
import os
import shutil
import sys  # Neu
import warnings

import gymnasium as gym
import numpy as np
import ray
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Deine Imports
from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy

# --- Logging & Warnings Setup ---
os.environ["RAY_DEDUP_LOGS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ray")

# --- Wrapper ---
class Float32ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=np.float32
        )

    def observation(self, observation):
        return np.array(observation, dtype=np.float32)

def env_creator(env_config):
    constants = env_config["constants"]
    env = TrainingEnv(constants)
    return Float32ObservationWrapper(env)

register_env("quantum_purify_env", env_creator)

# --- NEU: Globale Cleanup Funktion ---
def cleanup_ray():
    """Wird beim Beenden des Skripts aufgerufen."""
    if ray.is_initialized():
        print("\n!!! Führe Ray Shutdown durch (Aufräumen von RAM/CPU)... !!!")
        ray.shutdown()
        print("Ray Shutdown abgeschlossen.")

# Registriere die Funktion, damit sie auch bei Abstürzen läuft
atexit.register(cleanup_ray)

def main():
    # Temp-Ordner
    ray_temp_dir = os.path.abspath("./ray_temp")
    os.makedirs(ray_temp_dir, exist_ok=True)

    try:
        if not ray.is_initialized():
            print(f"Starte Ray (New API Stack Mode)...")
            ray.init(
                ignore_reinit_error=True, 
                _temp_dir=ray_temp_dir,
                include_dashboard=False, 
                log_to_driver=False
            )

        coherence_times = [0.001, 0.002, 0.003, 0.004, 0.005]
        
        for coherence_time in coherence_times:
            constants = ConstantsTuple(
                coherence_time=coherence_time,
                lambda_strategy=LambdaSrategy.USE_CONSTANTS,
                lambdas=(0.3, 0.0, 0.0),
                pumping_probability=1,
                waiting_time_sensitivity=1,
            )
            print(f"\n--- Training Start: Coherence Time {coherence_time} ---")
            run_name = f"{str(coherence_time).replace('.', '_')}"
            
            # Wir geben den Return-Wert zurück, um zu wissen, ob abgebrochen wurde
            was_interrupted = train(constants, run_name)
            
            if was_interrupted:
                print("Globaler Abbruch erkannt. Beende Hauptschleife.")
                break

    except KeyboardInterrupt:
        print("\n\n--- HAUPTPROGRAMM DURCH STRG+C UNTERBROCHEN ---")
    except Exception as e:
        print(f"Unerwarteter Fehler im Main: {e}")
    finally:
        # Das hier läuft IMMER, auch bei Fehler oder Strg+C
        cleanup_ray()


def train(constants: ConstantsTuple, run_name: str) -> bool:
    """
    Returns:
        bool: True wenn manuell abgebrochen (Strg+C), False wenn normal beendet.
    """
    log_dir = os.path.abspath("./ppo_results_rllib/")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config: AlgorithmConfig = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment(
            env="quantum_purify_env",
            env_config={"constants": constants},
        )
        .framework("torch")
        .env_runners(
            num_env_runners=4,
            num_envs_per_env_runner=1,
        )
        .training(
            lr=0.0001,
            gamma=1.0,
            lambda_=0.95,
            entropy_coeff=0.01,
            minibatch_size=128,
            num_epochs=10,
        )
        .rl_module(
            model_config={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                # "vf_share_layers": False,
            }
        )
        .evaluation(
            evaluation_interval=25,
            evaluation_duration=20,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )

    algo: Algorithm = config.build_algo()

    best_mean_reward = -float('inf')
    no_improvement_evals = 0
    max_no_improvement = 12
    min_evals_before_stop = 20
    eval_counter = 0
    
    total_timesteps = 50_000_000
    iterations = total_timesteps // 8192
    
    print(f"RLModule initialisiert. Starte Training...")
    
    interrupted = False # Flag um zu speichern ob abgebrochen wurde

    try:
        for i in range(iterations):
            result = algo.train()
            
            env_runner_metrics = result.get("env_runners", {})
            mean_reward = env_runner_metrics.get("episode_reward_mean", None)
            total_steps = result.get("num_env_steps_sampled", 0)

            if i % 5 == 0:
                reward_str = f"{mean_reward:.3f}" if mean_reward is not None else "Wait..."
                print(f"Iter: {i:4d} | Reward: {reward_str} | Total Steps: {total_steps}")

            # Evaluierung
            if "evaluation" in result and "env_runners" in result["evaluation"]:
                eval_metrics = result["evaluation"]["env_runners"]
                eval_mean_reward = eval_metrics.get("episode_reward_mean", None)
                
                if eval_mean_reward is None or np.isnan(eval_mean_reward):
                    continue

                eval_counter += 1
                print(f"   --> EVAL RESULT: {eval_mean_reward:.3f}")

                if eval_mean_reward > best_mean_reward:
                    best_mean_reward = eval_mean_reward
                    no_improvement_evals = 0
                    save_path = algo.save(checkpoint_dir)
                    print(f"   --> Neues Bestes Modell: {os.path.basename(save_path)}")
                else:
                    no_improvement_evals += 1
                    print(f"   --> Kein Fortschritt ({no_improvement_evals}/{max_no_improvement})")

                if eval_counter >= min_evals_before_stop and no_improvement_evals >= max_no_improvement:
                    print(f"Early Stopping! Iteration {i}")
                    break

    except KeyboardInterrupt:
        print("\nTraining manuell unterbrochen (in train() Loop)...")
        interrupted = True # Wir merken uns den Abbruch
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # WICHTIG: Algorithmus sauber stoppen um Worker freizugeben
        if algo:
            algo.stop()
        
        # Checkpoint speichern bei Abbruch
        final_path = algo.save(checkpoint_dir)
        print(f"Training beendet (Status gespeichert). Checkpoint: {final_path}")

    return interrupted

if __name__ == "__main__":
    main()