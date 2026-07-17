import numpy as np
import pytest

from ppo.policy_wrapper import canonicalize_obs, map_action_back
from purify.my_enums import Action

FOUR_ACTIONS = (Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3)


def make_obs(l1: float, l2: float, l3: float) -> np.ndarray:
    return np.array([0.7, 0.0, 0.1, l1, l2, l3], dtype=np.float64)


def test_canonicalize_sorts_lambdas_descending():
    obs = make_obs(0.05, 0.2, 0.05)
    canonical, order = canonicalize_obs(obs)

    assert list(canonical[3:6]) == [0.2, 0.05, 0.05]
    assert order[0] == 1  # größtes λ kam von Position 2 (Index 1)
    # Rest der Observation bleibt unangetastet
    assert list(canonical[:3]) == list(obs[:3])


def test_canonical_form_is_identity_for_training_distribution():
    """Für die Trainingsverteilung (Fehler auf λ₁) ändert sich nichts."""
    obs = make_obs(0.3, 0.0, 0.0)
    canonical, order = canonicalize_obs(obs)

    assert np.array_equal(canonical, obs)
    assert list(order) == [0, 1, 2]


def test_map_action_back_permutes_protocol():
    """Fehler auf λ₂ → Modell sieht kanonisch Fehler auf λ₁. Wählt es dort
    PROT_2 (wirksam gegen λ₁-Fehler), muss im Original-Raum das Protokoll
    herauskommen, das an der ursprünglichen λ-Anordnung dieselbe Wirkung hat."""
    obs = make_obs(0.0, 0.3, 0.0)
    _, order = canonicalize_obs(obs)
    # order = [1, 0, 2]: Kanonisch-Position 1 ← Original 2, Position 2 ← Original 1

    # Kanonische Wahl PROT_2 (Index 2) → Original: Protokoll der Original-Position order[1]=0
    assert map_action_back(2, order, FOUR_ACTIONS) == FOUR_ACTIONS.index(Action.PROT_1)
    # Kanonische Wahl PROT_1 (blind für den großen Fehler) → Original: PROT_2
    assert map_action_back(1, order, FOUR_ACTIONS) == FOUR_ACTIONS.index(Action.PROT_2)
    # Kanonische Wahl PROT_3 → Original: PROT_3 (Position unverändert)
    assert map_action_back(3, order, FOUR_ACTIONS) == FOUR_ACTIONS.index(Action.PROT_3)


def test_map_action_back_keeps_replace():
    obs = make_obs(0.0, 0.0, 0.3)
    _, order = canonicalize_obs(obs)
    assert map_action_back(0, order, FOUR_ACTIONS) == 0


def test_roundtrip_for_all_permutations():
    """Für jede Fehler-Position gilt: kanonische Wahl PROT_2/PROT_3 (die beiden
    wirksamen Protokolle) darf nie auf das blinde Protokoll der Original-
    Fehlerposition abgebildet werden."""
    for error_position in range(3):
        lambdas = [0.0, 0.0, 0.0]
        lambdas[error_position] = 0.3
        obs = make_obs(*lambdas)
        _, order = canonicalize_obs(obs)

        blind_protocol_idx = 1 + error_position  # PROT_{error_position+1}
        for canonical_choice in (2, 3):  # PROT_2, PROT_3 im kanonischen Raum
            mapped = map_action_back(canonical_choice, order, FOUR_ACTIONS)
            assert mapped != blind_protocol_idx


def test_symmetrized_physics_consistency():
    """End-to-End: Die zurückpermutierte Aktion muss physikalisch dieselbe
    Jump-Fidelity liefern wie die kanonische Aktion im kanonischen Zustand."""
    from purify.entanglement import Entanglement
    from purify.my_time import Time
    from purify.utils.purification_util import Purification

    time = Time(np.random.default_rng(0))

    def make_entanglement(fidelity, lambdas):
        return Entanglement(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=fidelity,
            creation_lambda_1=lambdas[0],
            creation_lambda_2=lambdas[1],
            creation_lambda_3=lambdas[2],
            decoherence_time=0.05,
        )

    e_good = make_entanglement(0.7, (0.1, 0.1, 0.1))

    obs = make_obs(0.05, 0.2, 0.05)
    canonical, order = canonicalize_obs(obs)

    e_bad_original = make_entanglement(0.7, tuple(obs[3:6]))
    e_bad_canonical = make_entanglement(0.7, tuple(canonical[3:6]))

    for canonical_choice in (1, 2, 3):
        mapped = map_action_back(canonical_choice, order, FOUR_ACTIONS)
        f_canonical = Purification.jump_function_from_action(
            e_good, e_bad_canonical, FOUR_ACTIONS[canonical_choice]
        )
        f_original = Purification.jump_function_from_action(
            e_good, e_bad_original, FOUR_ACTIONS[mapped]
        )
        assert f_original == pytest.approx(f_canonical)
