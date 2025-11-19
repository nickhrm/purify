from purify.my_enums import Strategy

ENTANGLEMENT_GENERATION_COUNT = 500000

# Times between bernouli trials for entanglement generation. in [sec]
# DELTA_T =   1.66782e-5

LENGTH = 5000
C = 2e8
DELTA_T = LENGTH/C
# Success probability of entangelement generation
P_G = 0.3047


QUBIT_ENTANGLEMENT_FACTOR = 100

QUBIT_ARRIVAL_SCALE = P_G/(QUBIT_ENTANGLEMENT_FACTOR * DELTA_T)


LAMBDA_1 = 0.0
LAMBDA_2 = 0.3
LAMBDA_3 = 0.0


# Stratgies to include in the simulation
STRATEGIES = [
        Strategy.ALWAYS_REPLACE,
        Strategy.ALWAYS_PROT_1,
        Strategy.ALWAYS_PROT_2,
        Strategy.ALWAYS_PROT_3,
    ]


# Decoherence times to include in the simulation in secs
DECOHERENCE_TIMES = [
        # 0.00001,
        0.0001,
        0.0003,
        0.0005,
        0.0008,
        0.001,
        0.003,
        0.005,
        0.008,
        0.01,
        0.05,
        0.1,
        1.0,
    ]

PUMPING_PROBABILTIES = [
    1
]

