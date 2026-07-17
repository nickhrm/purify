from typing import NamedTuple

from purify.my_enums import Action, LambdaStrategy


class ConstantsTuple(NamedTuple):
    coherence_time: float
    pumping_probability: float
    waiting_time_sensitivity: float
    lambda_strategy: LambdaStrategy

    # Nur relevant für LambdaStrategy.USE_CONSTANTS und RANDOM_WITH_LARGEST_LAMBDA:
    # Bei USE_CONSTANTS werden alle drei Werte direkt als λ₁, λ₂, λ₃ verwendet.
    # Bei RANDOM_WITH_LARGEST_LAMBDA wird nur lambdas[0] (λ₁) genutzt;
    # λ₂ und λ₃ werden zufällig aus dem verbleibenden Raum generiert.
    lambdas: tuple[float, float, float]

    actions: tuple[Action, ...]

    # Nur relevant für LambdaStrategy.RANDOM:
    # Definiert den Bereich, aus dem die initiale Fidelity gleichverteilt gezogen wird.
    min_fidelity: float
    max_fidelity: float

    def folder_name(self) -> str:
        lambda_parts = "_".join(str(lam).replace(".", "") for lam in self.lambdas)
        return f"{len(self.actions)}gps_{lambda_parts}"

    def subfolder_name(self) -> str:
        return f"{str(self.coherence_time).replace('.', '_')}"
