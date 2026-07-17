import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from ppo.case_studies import CASE_STUDIES
from ppo.custom_env import TrainingEnv


@pytest.fixture
def constants():
    return CASE_STUDIES[1].make_constants(coherence_time=0.01)


def test_env_passes_sb3_checker(constants):
    check_env(TrainingEnv(constants), warn=True)


def test_reset_returns_valid_observation_at_decision_point(constants):
    env = TrainingEnv(constants)
    obs, info = env.reset(seed=0)

    assert env.observation_space.contains(obs)
    # Am Entscheidungspunkt: Memory belegt und neues Entanglement vorhanden
    assert obs[0] > 0.0  # f_mem
    assert env.last_generated_entanglement is not None


def test_episodes_terminate_and_reward_is_fidelity(constants):
    env = TrainingEnv(constants)
    env.reset(seed=1)
    rng = np.random.default_rng(1)

    for _ in range(20):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0.0
        steps = 0
        while not terminated:
            action = int(rng.integers(env.action_space.n))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            assert not truncated
            assert steps < 100_000, "Episode terminiert nicht"

        # Reward = Teleportations-Fidelity des bedienten Requests
        assert 0.0 <= total_reward <= 1.0


def test_same_seed_gives_identical_trajectories(constants):
    def rollout(seed: int) -> list[float]:
        env = TrainingEnv(constants)
        obs, _ = env.reset(seed=seed)
        values = list(obs)
        for _ in range(200):
            obs, reward, terminated, truncated, _ = env.step(0)  # immer REPLACE
            values.extend(obs)
            values.append(reward)
            if terminated:
                obs, _ = env.reset()
        return values

    assert rollout(seed=123) == rollout(seed=123)
    assert rollout(seed=123) != rollout(seed=456)
