import os

import numpy as np
from ray.rllib.algorithms import AlgorithmConfig, PPOConfig

from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy


def train(constants: ConstantsTuple, run_name: str) -> bool:
    log_dir = os.path.abspath("./ppo_results_rllib/")
    checkpoint_dir = os.path.join(log_dir, "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_batch_size = 5000
    minibatch_size = 128
    num_env_runners = 6 # in old api stack: num_workers
    train_batch_size_per_learner = train_batch_size // num_env_runners


    total_timesteps = 50_000_000
    iterations = total_timesteps // train_batch_size

    config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            env=TrainingEnv,
            env_config={"constants": constants},
            disable_env_checking=False,
        )
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
            entropy_coeff=0.01,
            train_batch_size_per_learner=train_batch_size_per_learner,
            minibatch_size=minibatch_size,
            num_epochs=10,
            # model={
            #     "fcnet_hiddens": [64, 64],
            #     "fcnet_activation": "tanh",
            # },
        )
        .evaluation(
            evaluation_interval=25,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_config={"explore": False},
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )

    try:
        algo = config.build_algo()
    except Exception as e:
        print(f"Fehler beim Bauen des Algos: {e}")
        return True

    # Variablen für Evaluation
    best_mean_reward = -float("inf")
    no_improvement_evals = 0
    max_no_improvement = 12
    min_evals_before_stop = 20
    eval_counter = 0

    print(
        f"Algorithm initialisiert. Starte Training mit BatchSize {train_batch_size}..."
    )

    interrupted = False

    try:
        for i in range(iterations):
            result = algo.train()

            # Metriken extrahieren (kompatibel mit alter und neuer API)
            mean_reward = result.get("episode_reward_mean")
            if mean_reward is None and "env_runners" in result:
                mean_reward = result["env_runners"].get("episode_reward_mean")

            total_steps = result.get("num_env_steps_sampled", 0)

            # Logging
            if i % 1 == 0:
                reward_str = (
                    f"{mean_reward:.3f}" if mean_reward is not None else "Wait..."
                )
                print(
                    f"Iter: {i:4d} | Reward: {reward_str} | Total Steps: {total_steps}"
                )

            # --- Evaluation Logik ---
            if "evaluation" in result:
                # Pfad zu den Metriken finden
                eval_metrics = result["evaluation"]
                if "env_runners" in eval_metrics:
                    eval_metrics = eval_metrics["env_runners"]

                eval_mean_reward = eval_metrics.get("episode_reward_mean", None)

                if eval_mean_reward is not None and not np.isnan(eval_mean_reward):
                    eval_counter += 1
                    print(f"   --> EVAL RESULT: {eval_mean_reward:.3f}")

                    if eval_mean_reward > best_mean_reward:
                        best_mean_reward = eval_mean_reward
                        no_improvement_evals = 0
                        save_path = algo.save(checkpoint_dir)
                        print(
                            f"   --> Neues Bestes Modell: {os.path.basename(save_path)}"
                        )
                    else:
                        no_improvement_evals += 1
                        print(
                            f"   --> Kein Fortschritt ({no_improvement_evals}/{max_no_improvement})"  # noqa: E501
                        )

                    if (
                        eval_counter >= min_evals_before_stop
                        and no_improvement_evals >= max_no_improvement
                    ):
                        print(f"Early Stopping! Iteration {i}")
                        break

    except KeyboardInterrupt:
        print("\nTraining manuell unterbrochen (in train() Loop)...")
        interrupted = True  # Setze Flag auf True bei Abbruch

    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"Kritischer Fehler im Training: {e}")
        interrupted = True  # Auch bei Fehler als "unterbrochen" markieren

    finally:
        # Aufräumen und Speichern
        if algo:
            final_path = algo.save(checkpoint_dir)
            print(f"Training beendet. Letzter Checkpoint: {final_path}")
            algo.stop()

    # WICHTIG: Hier geben wir nun explizit den Status zurück
    return interrupted


def main():
    coherence_times = [0.001, 0.002, 0.003, 0.004, 0.005]
    # coherence_times = [0.006,  0.007, 0.008, 0.009, 0.010]
    # coherence_times = [0.020,  0.030, 0.040, 0.050, 0.060]
    # coherence_times = [0.070,  0.080, 0.090, 0.100,]

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
