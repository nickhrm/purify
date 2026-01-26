import os
import ray

import numpy as np
from ray.rllib.algorithms import Algorithm, AlgorithmConfig, PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from sympy.core.evalf import evalf_integer

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple, tupleAdapter
from purify.my_enums import LambdaSrategy


def train(constants: ConstantsTuple, run_name: str) -> bool:
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                "excludes": [
                    "**/.git/objects/pack/*.pack",
                ]
            }
        )
    log_dir = os.path.abspath("./ppo_results_rllib/")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_batch_size = 8000
    minibatch_size = 128
    num_env_runners = 6 # in old api stack: num_workers
    train_batch_size_per_learner = train_batch_size // num_env_runners
    eval_interval = 30

    total_timesteps = 50_000_000
    iterations = total_timesteps // train_batch_size

    config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            env=TrainingEnv,
            env_config=tupleAdapter(constants),
            disable_env_checking=False,
        )
        .rl_module(
        model_config=DefaultModelConfig(
            fcnet_hiddens=[64, 64],
            fcnet_activation="relu",
        ))
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            sample_timeout_s=60,
        )
        .training(
            lr=0.0001,
            gamma=1.0,
            lambda_=0.95,
            entropy_coeff=0.05,
            train_batch_size_per_learner=train_batch_size_per_learner,
            minibatch_size=minibatch_size,
            num_epochs=10,
        )
        .evaluation(
            evaluation_interval=eval_interval,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )

    algo: Algorithm = config.build_algo()

    # Variablen für Evaluation
    best_mean_reward = -float("inf")
    no_improvement_evals = 0
    max_no_improvement = 50
    min_evals_before_stop = 50
    eval_counter = 0

    print(
        f"Algorithm initialisiert. Starte Training mit BatchSize {train_batch_size}..."
    )


    interrupted = False

    try:
        for i in range(iterations):
            # ------------------------------------------------------------------
            # 1. TRAINING STEP
            # ------------------------------------------------------------------
            # algo.train() executes one training iteration.
            # In the new stack, this orchestrates the EnvRunners to sample
            # and the Learners to update weights.
            result = algo.train()

            # ------------------------------------------------------------------
            # 2. CHECKPOINTING (New API: save_to_path)
            # ------------------------------------------------------------------
            # We use save_to_path() which is the explicit, robust method for
            # persisting Algorithm state in RLlib 2.x+.
            # It returns the string path to the checkpoint directory.
            if i % 200 == 0:
                current_checkpoint_path = algo.save_to_path(checkpoint_dir)
                print(f"Checkpoint gespeichert: {current_checkpoint_path}")

            # ------------------------------------------------------------------
            # 3. METRIC EXTRACTION (Hierarchical Access)
            # ------------------------------------------------------------------
            # The result dictionary is now hierarchical.
            # "env_runners" -> Metrics from the environment sampling actors.
            # "learners"    -> Metrics from the gradient optimization actors.

            # Retrieve the 'env_runners' sub-dictionary.
            # Default to empty dict to prevent AttributeError on failure.
            env_runner_results = result.get("env_runners", {})

            # Extract 'episode_return_mean'.
            # 'episode_reward_mean' is deprecated. 'return' is the correct RL term.
            mean_ret = env_runner_results.get("episode_return_mean", None)

            # Extract total steps.
            # 'num_env_steps_sampled_lifetime' tracks total steps across all restarts.
            total_steps = result.get("num_env_steps_sampled_lifetime", 0)

            # ------------------------------------------------------------------
            # 4. LOGGING
            # ------------------------------------------------------------------
            if i % 1 == 0:
                # Handle potential None or NaN values during startup
                reward_str = (
                    f"{mean_ret:.3f}"
                    if mean_ret is not None and not np.isnan(mean_ret)
                    else "Wait..."
                )
                print(
                    f"Iter: {i:4d} | Return: {reward_str} | Total Steps: {total_steps}"
                )

            # ------------------------------------------------------------------
            # 5. EVALUATION LOGIC
            # ------------------------------------------------------------------
            # If evaluation_interval is set, RLlib runs evaluation automatically.
            # The results are stored under the "evaluation" key.
            if i > 0 and i % eval_interval == 0 and "evaluation" in result:
                eval_results = result["evaluation"]

                # Evaluation metrics also follow the 'env_runners' hierarchy
                eval_env_runner_results = eval_results.get("env_runners", {})
                eval_mean_ret = eval_env_runner_results.get("episode_return_mean", None)

                # Validate evaluation result
                if eval_mean_ret is not None and not np.isnan(eval_mean_ret):
                    eval_counter += 1
                    print(f"   --> EVAL RESULT: {eval_mean_ret:.3f}")

                    # ----------------------------------------------------------
                    # IMPROVEMENT CHECK & EARLY STOPPING
                    # ----------------------------------------------------------
                    if eval_mean_ret > best_mean_reward:
                        best_mean_reward = eval_mean_ret
                        no_improvement_evals = 0
                        # Save Best Model Explicitly
                        # We create a sub-folder to distinguish 'best' from periodic
                        best_model_dir = os.path.join(checkpoint_dir, "best_model")
                        os.makedirs(best_model_dir, exist_ok=True)
                        # Save the best model
                        save_path = algo.save_to_path(best_model_dir)
                        print(
                            f"   --> Neues Bestes Modell: {os.path.basename(save_path)} for t_c = {constants.coherence_time}"
                        )
                    else:
                        no_improvement_evals += 1
                        print(
                            f"   --> Kein Fortschritt ({no_improvement_evals}/{max_no_improvement}) for t_c = {constants.coherence_time}"
                        )

                    # Early Stopping Condition
                    if (
                        eval_counter >= min_evals_before_stop
                        and no_improvement_evals >= max_no_improvement
                    ):
                        print(f"Early Stopping! Iteration {i}")
                        break

    except KeyboardInterrupt:
        print("\nTraining manuell unterbrochen (in train() Loop)...")
        interrupted = True  # Set flag on manual interrupt

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Kritischer Fehler im Training: {e}")
        interrupted = True  # Set flag on error

    finally:
        # ------------------------------------------------------------------
        # 6. CLEANUP & FINAL SAVE
        # ------------------------------------------------------------------
        if algo:
            # Ensure the final state is saved before teardown
            final_path = algo.save_to_path(checkpoint_dir)
            print(f"Training beendet. Letzter Checkpoint: {final_path}")
            algo.stop()

    # Explicit return of the interrupted status
    return interrupted

def main():
    # coherence_times = [0.001, 0.002, 0.003, 0.004, 0.005]
    # coherence_times = [0.006,  0.007, 0.008, 0.009, 0.010]
    # coherence_times = [0.020,  0.030, 0.040, 0.050, 0.060]
    coherence_times = [0.070,  0.080, 0.090, 0.100,]

    # lambdas = [
    #     (0.3, 0.0, 0.0),
    #     (0.0, 0.3, 0.0),
    #     (0.0, 0.0, 0.3),
    # ]

    for coherence_time in coherence_times:
        # # First train for fixed lambdas
        # print(f"Training coherence_time: {coherence_time}")
        # for lam in lambdas:
        # print(f"Training with lambda: {lam}")
        #     constants = ConstantsTuple(
        #         coherence_time=coherence_time,
        #         lambda_strategy=LambdaSrategy.USE_CONSTANTS,
        #         lambdas=lam,
        #         pumping_probability=1,
        #         waiting_time_sensitivity=1,
        #     )
        #     run_name = f"{str(coherence_time).replace(".","_")}"
        #     train(constants, run_name)

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
