from enum import Enum


class Event(Enum):
    ENTANGLEMENT_GENERATION = 1
    REQUEST_ARRIVAL = 2


class Action(Enum):
    REPLACE = 0
    PROT_1 = 1
    PROT_2 = 2
    PROT_3 = 3
    PMD = 4


class LambdaStrategy(Enum):
    USE_CONSTANTS = 1
    RANDOM_WITH_LARGEST_LAMBDA = 2
    RANDOM = 3
    FIXED = 4
    # Feste Lambda-Werte aus den Constants, aber pro Entanglement zufällig
    # auf die Positionen λ₁, λ₂, λ₃ permutiert. Trainings-Augmentierung für
    # bessere Generalisierung auf beliebige Fehler-Positionen.
    PERMUTED_CONSTANTS = 5
