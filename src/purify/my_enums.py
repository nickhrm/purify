from enum import Enum


class Event(Enum):
    ENTANGLEMENT_GENERATION = 1
    REQUEST_ARRIVAL = 2


class Action(Enum):
    REPLACE = 0
    PROT_1 = 1
    PROT_2 = 2
    PROT_3 = 3
    PROT_1_WITH_PROBABILITY = 4
    PROT_2_WITH_PROBABILITY = 5
    PROT_3_WITH_PROBABILITY = 6
    CHOOSE_RANDOM_ACTION_UNIFORMLY = 8
    GPS = 9
    ALWAYS_PMD = 10
    TRAINING_MODE = 11


class LambdaSrategy(Enum):
    USE_CONSTANTS = 1
    RANDOM_WITH_LARGEST_LAMBDA = 2

