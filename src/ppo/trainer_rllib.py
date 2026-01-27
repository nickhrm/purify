import os
import shutil
import time
import numpy as np
import ray
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPO

# Deine Imports
from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple, tupleAdapter
from purify.my_enums import LambdaSrategy

def env_creator(env_config):
    return TrainingEnv(env_config)

def train(constants: ConstantsTuple, run_name: str) -> bool:
    # Ray initialisieren, falls noch nicht geschehen
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "excludes": ["**/.git/objects/pack/*.pack"]
            },
            _temp_dir=os.path.expanduser("~/ray_tmp")
        )

    # Environment registrieren
    register_env("purify_env", env_creator)

    # Pfade
    log_dir = os.path.abspath("./ppo_results_rllib/")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", run_name)
    best_model_dir = os.path.join(checkpoint_dir, "best_model")
    
    # Sicherstellen, dass Verzeichnisse existieren
    if os.path.exists(checkpoint_dir):
        # Optional: Reinigen, wenn man von null starten will. 
        # Hier behalten wir es mal oder machen es neu, je nach Wunsch.
        pass
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # Hyperparameter (entsprechend deinem vorherigen Config-Objekt)
    train_batch_size = 8192
    num_env_runners = 4
    minibatch_size = 128
    
    config = {
        "env": "purify_env",
        "env_config": tupleAdapter(constants),
        "framework": "torch",
        "num_workers": num_env_runners,
        "num_envs_per_worker": 1,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": minibatch_size,
        "num_sgd_iter": 10,
        "lr": 0.0001,
        "gamma": 1.0,
        "lambda": 0.95,
        "entropy_coeff": 0.01,
        "clip_param": 0.2,
        "vf_loss_coeff": 0.5,
        "grad_clip": 0.5,
        "kl_coeff": 0.0,
        "vf_clip_param": 100.0,
        # Evaluation config
        "evaluation_interval": 25,
        "evaluation_duration": 10,
        "evaluation_duration_unit": "episodes",
        "evaluation_config": {
            "explore": False
        },
        "num_gpus": 0,
        "log_level": "ERROR",
    }

    print(f"Initialisiere PPO Trainer für {run_name}...")
    # Trainer Instanz erstellen (Legacy Style Init)
    trainer = PPO(config=config)

    # Variablen für den Loop
    total_timesteps = 50_000_000
    iterations = total_timesteps // train_batch_size
    
    best_mean_reward = -float("inf")
    no_improvement_evals = 0
    max_no_improvement = 50
    min_evals_before_stop = 50
    eval_counter = 0

    interrupted = False
    
    print(f"Starte Training: {iterations} Iterationen...")

    try:
        for i in range(iterations):
            # 1. Train step
            result = trainer.train()
            
            # Wichtige Metriken extrahieren
            # Hinweis: In neueren Versionen ist die Struktur result["env_runners"]["episode_return_mean"]
            # In älteren Versionen war es result["episode_reward_mean"]
            # Wir versuchen, beides sicher abzufangen.
            if "env_runners" in result:
                mean_ret = result["env_runners"].get("episode_return_mean", None)
            else:
                mean_ret = result.get("episode_reward_mean", None)
            
            total_steps = result.get("num_env_steps_sampled_lifetime", 0)
            if total_steps == 0:
                total_steps = result.get("timesteps_total", 0)

            # Logging
            if i % 1 == 0:
                reward_str = f"{mean_ret:.3f}" if mean_ret is not None else "Wait..."
                print(f"Iter: {i:4d} | Return: {reward_str} | Total Steps: {total_steps}")

            # 2. Checkpoint speichern (periodisch)
            if i % 200 == 0:
                ckpt = trainer.save(checkpoint_dir)
                print(f"Checkpoint gespeichert: {ckpt}")

            # 3. Evaluation Handling
            # Wenn evaluation_interval in config gesetzt ist, führt .train() das schon aus.
            # Die Ergebnisse sind in result["evaluation"]
            if "evaluation" in result:
                eval_metrics = result["evaluation"]
                # Auch hier wieder Struktur checken
                if "env_runners" in eval_metrics:
                    eval_mean_ret = eval_metrics["env_runners"].get("episode_return_mean", None)
                else:
                    eval_mean_ret = eval_metrics.get("episode_reward_mean", None)
                
                if eval_mean_ret is not None and not np.isnan(eval_mean_ret):
                    eval_counter += 1
                    print(f"   --> EVAL RESULT: {eval_mean_ret:.3f}")

                    if eval_mean_ret > best_mean_reward:
                        best_mean_reward = eval_mean_ret
                        no_improvement_evals = 0
                        
                        # Speichere bestes Modell
                        ckpt = trainer.save(best_model_dir)
                        print(f"   --> Neues Bestes Modell: {os.path.basename(ckpt)}")
                    else:
                        no_improvement_evals += 1
                        print(f"   --> Kein Fortschritt ({no_improvement_evals}/{max_no_improvement})")

                    # Early Stopping
                    if eval_counter >= min_evals_before_stop and no_improvement_evals >= max_no_improvement:
                        print(f"Early Stopping! Iteration {i}")
                        break

    except KeyboardInterrupt:
        print("\nTraining manuell unterbrochen.")
        interrupted = True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fehler: {e}")
        interrupted = True
    finally:
        # Cleanup
        final_ckpt = trainer.save(checkpoint_dir)
        print(f"Training beendet. Finaler Checkpoint: {final_ckpt}")
        trainer.stop()
    
    return interrupted


def main():
    coherence_times = [0.100]

    for coherence_time in coherence_times:
        constants = ConstantsTuple(
            coherence_time=coherence_time,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            pumping_probability=1,
            waiting_time_sensitivity=1,
        )
        print(f"Training Randomly for coherence time: {coherence_time}")
        run_name = f"{str(coherence_time).replace('.', '_')}"
        train(constants, run_name)

if __name__ == "__main__":
    main()
