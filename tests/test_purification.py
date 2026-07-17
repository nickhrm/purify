import numpy as np
import pytest

from purify.entanglement import Entanglement
from purify.my_enums import Action
from purify.my_time import Time
from purify.utils.purification_util import Purification


def make_entanglement(
    time: Time,
    fidelity: float,
    lambdas: tuple[float, float, float],
    coherence_time: float = 0.05,
) -> Entanglement:
    return Entanglement(
        time=time,
        creation_time=time.get_current_time(),
        creation_fidelity=fidelity,
        creation_lambda_1=lambdas[0],
        creation_lambda_2=lambdas[1],
        creation_lambda_3=lambdas[2],
        decoherence_time=coherence_time,
    )


@pytest.fixture
def time() -> Time:
    return Time(np.random.default_rng(0))


def test_purification_improves_fidelity_for_suitable_protocol(time):
    """Fehler nur auf λ₁: PROT_2 und PROT_3 müssen die Fidelity erhöhen,
    während PROT_1 für Fehler auf λ₁ blind ist und sie sogar verschlechtert."""
    e_good = make_entanglement(time, 0.7, (0.1, 0.1, 0.1))  # Werner
    e_bad = make_entanglement(time, 0.7, (0.3, 0.0, 0.0))

    f_prot_2 = Purification.jump_function_from_action(e_good, e_bad, Action.PROT_2)
    f_prot_3 = Purification.jump_function_from_action(e_good, e_bad, Action.PROT_3)
    f_prot_1 = Purification.jump_function_from_action(e_good, e_bad, Action.PROT_1)

    assert 0.7 < f_prot_2 <= 1.0
    assert 0.7 < f_prot_3 <= 1.0
    assert f_prot_1 < 0.7  # blind für den eigenen Fehlerkanal


def test_success_probability_in_valid_range(time):
    e_good = make_entanglement(time, 0.7, (0.1, 0.1, 0.1))
    e_bad = make_entanglement(time, 0.7, (0.3, 0.0, 0.0))

    for action in (Action.PROT_1, Action.PROT_2, Action.PROT_3):
        p = Purification.success_probability_from_action(e_good, e_bad, action)
        assert 0.0 <= p <= 1.0


def test_success_probability_zero_below_half_fidelity(time):
    """Unterhalb von F = 0.5 ist Purification nicht möglich."""
    e_good = make_entanglement(time, 0.5, (0.2, 0.2, 0.1))
    e_bad = make_entanglement(time, 0.7, (0.3, 0.0, 0.0))

    assert Purification.success_probability_from_action(e_good, e_bad, Action.PROT_1) == 0.0
    assert Purification.success_probability_from_action(e_bad, e_good, Action.PROT_1) == 0.0


def test_protocols_are_symmetric_under_lambda_permutation(time):
    """PROT_k adressiert λ_k: Fehler auf λ₁ mit PROT_1 muss dasselbe Ergebnis
    liefern wie Fehler auf λ₂ mit PROT_2 bzw. λ₃ mit PROT_3."""
    e_good = make_entanglement(time, 0.7, (0.1, 0.1, 0.1))

    cases = [
        (Action.PROT_1, (0.3, 0.0, 0.0)),
        (Action.PROT_2, (0.0, 0.3, 0.0)),
        (Action.PROT_3, (0.0, 0.0, 0.3)),
    ]
    fidelities = []
    probabilities = []
    for action, lambdas in cases:
        e_bad = make_entanglement(time, 0.7, lambdas)
        fidelities.append(Purification.jump_function_from_action(e_good, e_bad, action))
        probabilities.append(Purification.success_probability_from_action(e_good, e_bad, action))

    assert fidelities[0] == pytest.approx(fidelities[1])
    assert fidelities[0] == pytest.approx(fidelities[2])
    assert probabilities[0] == pytest.approx(probabilities[1])
    assert probabilities[0] == pytest.approx(probabilities[2])


def test_pmd_requires_lambda_2_and_3_zero(time):
    e_good = make_entanglement(time, 0.7, (0.1, 0.1, 0.1))
    e_bad = make_entanglement(time, 0.7, (0.2, 0.1, 0.0))

    with pytest.raises(Exception):
        Purification.jump_function_from_action(e_good, e_bad, Action.PMD)


def test_pmd_matches_closed_form(time):
    e_good = make_entanglement(time, 0.8, (0.2 / 3, 0.2 / 3, 0.2 / 3))
    e_bad = make_entanglement(time, 0.7, (0.3, 0.0, 0.0))

    f_g, f_b = 0.8, 0.7
    expected_p = f_b * f_g + (1 - f_b) * (1 - f_g)
    expected_f = f_b * f_g / expected_p

    p = Purification.success_probability_from_action(e_good, e_bad, Action.PMD)
    f = Purification.jump_function_from_action(e_good, e_bad, Action.PMD)
    assert p == pytest.approx(expected_p)
    assert f == pytest.approx(expected_f)
