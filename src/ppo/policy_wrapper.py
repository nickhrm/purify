from typing import Any

import numpy as np
from stable_baselines3 import PPO

from purify.my_enums import Action

# Reihenfolge entspricht den λ-Positionen: PROT_k gehört zu λ_k
PROTOCOL_ACTIONS = (Action.PROT_1, Action.PROT_2, Action.PROT_3)

# Observation-Layout: [f_mem, request_is_waiting, time_since_last_request, l1, l2, l3]
_LAMBDA_SLICE = slice(3, 6)


def canonicalize_obs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sortiert die λ der Observation absteigend (größter Fehler auf Position 1).

    Returns (canonical_obs, order), wobei order[k] die Original-Position ist,
    die in der kanonischen Form auf Position k liegt.
    """
    lambdas = np.asarray(obs[_LAMBDA_SLICE])
    order = np.argsort(-lambdas, kind="stable")
    canonical_obs = np.array(obs, copy=True)
    canonical_obs[_LAMBDA_SLICE] = lambdas[order]
    return canonical_obs, order


def map_action_back(action_idx: int, order: np.ndarray, actions: tuple[Action, ...]) -> int:
    """Übersetzt eine im kanonischen Raum gewählte Aktion zurück in den
    Original-Raum: PROT_k (kanonische Position k) wird zum Protokoll der
    Original-Position order[k]; alle anderen Aktionen bleiben unverändert."""
    action = actions[action_idx]
    if action not in PROTOCOL_ACTIONS:
        return action_idx  # REPLACE / PMD sind permutations-invariant

    canonical_position = PROTOCOL_ACTIONS.index(action)
    original_position = int(order[canonical_position])
    mapped_action = PROTOCOL_ACTIONS[original_position]
    return actions.index(mapped_action)


class PolicyWrapper:
    def __init__(self, name):
        self.name = name

    def predict(self, obs) -> Any:
        raise NotImplementedError


class SB3Agent(PolicyWrapper):
    def __init__(self, path, env, deterministic: bool = True):
        super().__init__("PPO AI")
        self.model = PPO.load(path, env=env, device="cpu")
        self.deterministic = deterministic

    def predict(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return action


class SymmetrizedSB3Agent(SB3Agent):
    """Macht eine Policy permutations-äquivariant, ohne sie neu zu trainieren.

    Die Physik ist invariant unter simultaner Permutation von (λ₁, λ₂, λ₃)
    und (PROT_1, PROT_2, PROT_3). Ein Modell, das nur Fehler auf λ₁ gesehen
    hat (z.B. λ=(0.3, 0, 0)), bekommt hier die Observation in kanonischer
    Form (λ absteigend sortiert – größter Fehler auf Position 1 wie im
    Training). Die gewählte PROT-Aktion wird anschließend durch die inverse
    Permutation zurück auf die tatsächlichen λ-Positionen übersetzt.
    """

    def __init__(self, path, env, actions: tuple[Action, ...], deterministic: bool = True):
        super().__init__(path, env, deterministic)
        self.name = "PPO AI (symmetrized)"
        if not all(protocol in actions for protocol in PROTOCOL_ACTIONS):
            raise ValueError(
                "Symmetrisierung braucht alle drei Protokolle PROT_1..3 im "
                f"Aktionsraum, vorhanden: {[a.name for a in actions]}"
            )
        self.actions = actions

    def predict(self, obs):
        canonical_obs, order = canonicalize_obs(obs)
        action_idx = int(super().predict(canonical_obs))
        return map_action_back(action_idx, order, self.actions)


class FixedActionAgent(PolicyWrapper):
    def __init__(self, action: Action):
        super().__init__(f"Always {action.name}")
        self.action = action.value

    def predict(self, obs):
        return self.action


class RandomAgent(PolicyWrapper):
    def __init__(self, action_space):
        super().__init__("Random")
        self.aspace = action_space

    def predict(self, obs):
        return self.aspace.sample()
