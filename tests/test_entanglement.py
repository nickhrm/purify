import numpy as np
import pytest

from purify.constants_tuple import ConstantsTuple
from purify.entanglement import Entanglement
from purify.my_constants import DELTA_T
from purify.my_enums import Action, LambdaStrategy
from purify.my_time import Time


def make_constants(
    lambda_strategy: LambdaStrategy,
    lambdas: tuple[float, float, float] = (0.3, 0.0, 0.0),
    min_fidelity: float = 0.7,
    max_fidelity: float = 0.7,
) -> ConstantsTuple:
    return ConstantsTuple(
        coherence_time=0.05,
        pumping_probability=1,
        waiting_time_sensitivity=1,
        lambda_strategy=lambda_strategy,
        lambdas=lambdas,
        actions=(Action.REPLACE,),
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
    )


def test_fidelity_equals_creation_value_at_creation_time():
    time = Time(np.random.default_rng(0))
    e = Entanglement.from_fidelity(time, 0.9, decoherence_time=0.05)
    assert e.get_current_fidelity() == pytest.approx(0.9)


def test_fidelity_decays_towards_one_quarter():
    rng = np.random.default_rng(0)
    time = Time(rng)
    e = Entanglement.from_fidelity(time, 0.9, decoherence_time=0.05)

    previous = e.get_current_fidelity()
    for _ in range(2000):
        time.update()
        current = e.get_current_fidelity()
        assert current <= previous
        previous = current

    # Nach vielen Kohärenzzeiten: maximal gemischter Zustand
    assert previous >= 0.25


def test_decay_preserves_normalization():
    """F + λ₁ + λ₂ + λ₃ muss zu jedem Zeitpunkt 1 bleiben."""
    time = Time(np.random.default_rng(0))
    constants = make_constants(LambdaStrategy.USE_CONSTANTS, lambdas=(0.2, 0.05, 0.05))
    e = Entanglement.from_strategy(time, constants)

    for _ in range(500):
        time.update()

    total = (
        e.get_current_fidelity()
        + e.get_current_lambda_1()
        + e.get_current_lambda_2()
        + e.get_current_lambda_3()
    )
    assert total == pytest.approx(1.0)


def test_from_fidelity_rejects_invalid_values():
    time = Time(np.random.default_rng(0))
    with pytest.raises(ValueError):
        Entanglement.from_fidelity(time, 1.2, decoherence_time=0.05)
    with pytest.raises(ValueError):
        Entanglement.from_fidelity(time, -0.1, decoherence_time=0.05)


def test_use_constants_strategy_sets_lambdas_exactly():
    time = Time(np.random.default_rng(0))
    constants = make_constants(LambdaStrategy.USE_CONSTANTS, lambdas=(0.3, 0.0, 0.0))
    e = Entanglement.from_strategy(time, constants)

    assert e.creation_fidelity == pytest.approx(0.7)
    assert e.creation_lambda_1 == pytest.approx(0.3)
    assert e.creation_lambda_2 == 0.0
    assert e.creation_lambda_3 == 0.0


def test_random_strategy_distributes_error_mass():
    """RANDOM mit min=max=0.7: F fest, Fehlermasse 0.3 zufällig auf λ₁..λ₃."""
    rng = np.random.default_rng(42)
    time = Time(rng)
    constants = make_constants(LambdaStrategy.RANDOM)

    for _ in range(200):
        e = Entanglement.from_strategy(time, constants, rng)
        assert e.creation_fidelity == pytest.approx(0.7)
        lambdas = (e.creation_lambda_1, e.creation_lambda_2, e.creation_lambda_3)
        assert all(lam >= 0 for lam in lambdas)
        assert sum(lambdas) == pytest.approx(0.3)


def test_permuted_constants_strategy_shuffles_error_position():
    """PERMUTED_CONSTANTS: Werte bleiben (0.3, 0, 0), aber die Fehler-Position
    variiert über alle drei λ."""
    rng = np.random.default_rng(42)
    time = Time(rng)
    constants = make_constants(LambdaStrategy.PERMUTED_CONSTANTS, lambdas=(0.3, 0.0, 0.0))

    seen_positions = set()
    for _ in range(100):
        e = Entanglement.from_strategy(time, constants, rng)
        lambdas = (e.creation_lambda_1, e.creation_lambda_2, e.creation_lambda_3)
        assert sorted(lambdas, reverse=True) == pytest.approx([0.3, 0.0, 0.0])
        assert e.creation_fidelity == pytest.approx(0.7)
        seen_positions.add(int(np.argmax(lambdas)))

    assert seen_positions == {0, 1, 2}


def test_random_strategy_is_reproducible_with_seed():
    time = Time(np.random.default_rng(0))
    constants = make_constants(LambdaStrategy.RANDOM)

    e1 = Entanglement.from_strategy(time, constants, np.random.default_rng(7))
    e2 = Entanglement.from_strategy(time, constants, np.random.default_rng(7))
    assert e1.creation_lambda_1 == e2.creation_lambda_1
    assert e1.creation_lambda_2 == e2.creation_lambda_2
    assert e1.creation_lambda_3 == e2.creation_lambda_3


def test_time_advances_monotonically_and_reproducibly():
    t1 = Time(np.random.default_rng(3))
    t2 = Time(np.random.default_rng(3))

    previous = 0.0
    for _ in range(1000):
        t1.update()
        t2.update()
        assert t1.get_current_time() >= previous
        assert t1.get_current_time() == t2.get_current_time()
        assert t1.last_event() == t2.last_event()
        previous = t1.get_current_time()

    # Entanglement-Events kommen in festem DELTA_T-Raster
    assert t1.entanglement_time == pytest.approx(t1.entanglement_count * DELTA_T)
