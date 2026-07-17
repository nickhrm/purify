from dataclasses import dataclass, replace

from ppo.hyperparams_tuple import HyperparamsTuple
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaStrategy


@dataclass
class CaseStudy:
    """
    Definiert eine Case Study: eine festgelegte Umgebungskonfiguration (Constants)
    und eine Liste zu trainierender Kohärenzzeiten.

    Die eigentliche coherence_time wird erst beim Trainingsstart per
    `make_constants()` in das ConstantsTuple eingesetzt. Welche Kohärenzzeiten
    auf einem Gerät trainiert werden, wird über die CLI ausgewählt
    (siehe ppo.trainer); `coherence_times` ist der Default.
    """

    id: int
    notes: str
    constants: ConstantsTuple
    hyperparams: HyperparamsTuple
    coherence_times: list[float]
    deterministic: bool = True
    waiting_time_sensitivities: list[float] | None = None

    def make_constants(self, coherence_time: float) -> ConstantsTuple:
        """Gibt ein neues ConstantsTuple zurück, bei dem coherence_time gesetzt ist."""
        return self.constants._replace(coherence_time=coherence_time)

    def make_constants_wts(self, coherence_time: float, wt_sense: float) -> ConstantsTuple:
        """Gibt ein neues ConstantsTuple zurück, bei dem coherence_time UND
        waiting_time_sensitivity gesetzt sind."""
        return self.constants._replace(
            coherence_time=coherence_time,
            waiting_time_sensitivity=wt_sense,
        )

    def folder_name(self) -> str:
        return f"case_study_{self.id}"

    @staticmethod
    def coherence_time_str(coherence_time: float) -> str:
        """Erzeugt einen dateisystemfreundlichen Ordnernamen für eine Kohärenzzeit."""
        return f"T{str(coherence_time).replace('.', '_')}"

    @staticmethod
    def wts_str(wt_sense: float) -> str:
        """Erzeugt einen dateisystemfreundlichen Ordnernamen für eine waiting_time_sensitivity."""
        return f"wts_{str(wt_sense).replace('.', '_')}"


# ─── Gemeinsame Defaults ──────────────────────────────────────────────────────

# Voller untersuchter Bereich: 0.001–0.009 (fein) und 0.01–0.1 (grob).
CT_FINE: list[float] = [round(i * 0.001, 3) for i in range(1, 10)]
CT_COARSE: list[float] = [round(i * 0.01, 2) for i in range(1, 11)]
CT_ALL: list[float] = CT_FINE + CT_COARSE

FOUR_PROTOCOLS = (Action.REPLACE, Action.PROT_1, Action.PROT_2, Action.PROT_3)
FOUR_PROTOCOLS_PMD = FOUR_PROTOCOLS + (Action.PMD,)


def make_study_constants(
    lambdas: tuple[float, float, float] = (0.3, 0.0, 0.0),
    lambda_strategy: LambdaStrategy = LambdaStrategy.USE_CONSTANTS,
    actions: tuple[Action, ...] = FOUR_PROTOCOLS,
) -> ConstantsTuple:
    """Constants-Vorlage aller Case Studies; nur die genannten Felder variieren."""
    return ConstantsTuple(
        coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
        pumping_probability=1,
        waiting_time_sensitivity=1,
        lambda_strategy=lambda_strategy,
        lambdas=lambdas,
        actions=actions,
        min_fidelity=0.7,
        max_fidelity=0.7,
    )


STANDARD_HYPERPARAMS = HyperparamsTuple(
    n_steps=1024,
    batch_size=256,
    n_epochs=6,
    learning_rate=2.9496e-4,
    gamma=1.0,
    gae_lambda=0.99,
    ent_coef=0.003011806764086755,
    clip_range=0.2,
    vf_coef=0.5,
    max_grad_norm=0.5,
    pi_layers=[256, 256],
    vf_layers=[256, 256],
)

# Optuna-Ergebnis für die PMD-Studien (CS 5, 15, 20)
PMD_TUNED_HYPERPARAMS = replace(
    STANDARD_HYPERPARAMS,
    n_steps=256,
    batch_size=128,
    n_epochs=5,
    learning_rate=4.310293e-04,
    gae_lambda=0.939620,
    ent_coef=8.393195e-04,
)


# ─── Case Studies ─────────────────────────────────────────────────────────────
# Jede Case Study hat eine eindeutige ID; sie entspricht dem Ordner
# results/case_study_{id}. IDs bereits trainierter Studien nicht recyceln.
# ─────────────────────────────────────────────────────────────────────────────

CASE_STUDIES: dict[int, CaseStudy] = {
    1: CaseStudy(
        id=1,
        notes="Standard Entanglement, λ-Fehler auf λ₁",
        constants=make_study_constants(lambdas=(0.3, 0.0, 0.0)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    2: CaseStudy(
        id=2,
        notes="Standard Entanglement, λ-Fehler auf λ₂",
        constants=make_study_constants(lambdas=(0.0, 0.3, 0.0)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    3: CaseStudy(
        id=3,
        notes="Standard Entanglement, λ-Fehler auf λ₃",
        constants=make_study_constants(lambdas=(0.0, 0.0, 0.3)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    4: CaseStudy(
        id=4,
        notes="4-GPS, Fehler zufällig verteilt (RANDOM)",
        constants=make_study_constants(
            lambdas=(0.0, 0.0, 0.0), lambda_strategy=LambdaStrategy.RANDOM
        ),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    5: CaseStudy(
        id=5,
        notes="Mit PMD",
        constants=make_study_constants(actions=FOUR_PROTOCOLS_PMD),
        hyperparams=PMD_TUNED_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    6: CaseStudy(
        id=6,
        notes="lambdas 0.2, 0.1, 0.0",
        constants=make_study_constants(lambdas=(0.2, 0.1, 0.0)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    7: CaseStudy(
        id=7,
        notes="Neue Hyperparameter ausprobieren",
        constants=make_study_constants(),
        hyperparams=replace(
            STANDARD_HYPERPARAMS,
            n_steps=512,
            learning_rate=3.283142e-04,
            gae_lambda=0.953157,
            ent_coef=1.713062e-02,
        ),
        coherence_times=[0.08],
    ),
    8: CaseStudy(
        id=8,
        notes="Standard Entanglement, Training für kleine coherence times "
        "mit Parametern von 0.008-Optuna",
        constants=make_study_constants(),
        hyperparams=replace(
            STANDARD_HYPERPARAMS,
            n_steps=2048,
            n_epochs=9,
            learning_rate=3.55006612192314e-05,
            ent_coef=0.4962618370958482,
            pi_layers=[128, 128],
            vf_layers=[128, 128],
        ),
        coherence_times=[0.008, 0.01],
    ),
    9: CaseStudy(
        id=9,
        notes="Standard Entanglement (wie CS 1)",
        constants=make_study_constants(),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    10: CaseStudy(
        id=10,
        notes="4-GPS Random (wie CS 4), kleine Kohärenzzeiten",
        constants=make_study_constants(
            lambdas=(0.0, 0.0, 0.0), lambda_strategy=LambdaStrategy.RANDOM
        ),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_FINE + [0.01],
    ),
    11: CaseStudy(
        id=11,
        notes="λ-Fehler auf λ₂ (wie CS 2), kleine Kohärenzzeiten",
        constants=make_study_constants(lambdas=(0.0, 0.3, 0.0)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_FINE + [0.01],
    ),
    12: CaseStudy(
        id=12,
        notes="λ-Fehler auf λ₃ (wie CS 3), kleine Kohärenzzeiten",
        constants=make_study_constants(lambdas=(0.0, 0.0, 0.3)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_FINE + [0.01],
    ),
    13: CaseStudy(
        id=13,
        notes="2-GPS PMD, nicht deterministisch",
        constants=make_study_constants(actions=(Action.REPLACE, Action.PMD)),
        hyperparams=replace(
            STANDARD_HYPERPARAMS,
            n_steps=256,
            batch_size=1024,
            n_epochs=13,
            learning_rate=3.041823e-04,
            gae_lambda=0.940096,
            ent_coef=1.303879e-07,
        ),
        coherence_times=CT_ALL,
    ),
    14: CaseStudy(
        id=14,
        notes="Mit PMD, alternative Optuna-Parameter",
        constants=make_study_constants(actions=FOUR_PROTOCOLS_PMD),
        hyperparams=replace(
            STANDARD_HYPERPARAMS,
            n_steps=512,
            batch_size=128,
            n_epochs=15,
            learning_rate=4.721431e-04,
            gae_lambda=0.938491,
            ent_coef=1.375055e-07,
        ),
        coherence_times=CT_ALL,
    ),
    15: CaseStudy(
        id=15,
        notes="5-GPS nicht-deterministisch, params von CS 5",
        constants=make_study_constants(actions=FOUR_PROTOCOLS_PMD),
        deterministic=True,
        hyperparams=PMD_TUNED_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    17: CaseStudy(
        id=17,
        notes="5-GPS mit Standard-Hyperparametern",
        constants=make_study_constants(actions=FOUR_PROTOCOLS_PMD),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    18: CaseStudy(
        id=18,
        notes="2-GPS PMD mit Standard-Hyperparametern",
        constants=make_study_constants(actions=(Action.REPLACE, Action.PMD)),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    19: CaseStudy(
        id=19,
        notes="2-GPS ohne PMD",
        constants=make_study_constants(actions=(Action.REPLACE, Action.PROT_2)),
        deterministic=False,
        hyperparams=replace(
            STANDARD_HYPERPARAMS,
            batch_size=1024,
            n_epochs=5,
            learning_rate=1.067748e-04,
            gae_lambda=0.959241,
            ent_coef=2.114233e-08,
        ),
        coherence_times=CT_ALL,
    ),
    20: CaseStudy(
        id=20,
        notes="2-GPS PMD, nicht deterministisch, Parameter für mittlere Kohärenzzeiten",
        constants=make_study_constants(actions=(Action.REPLACE, Action.PMD)),
        deterministic=False,
        hyperparams=PMD_TUNED_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
    21: CaseStudy(
        id=21,
        notes="Generalisierungs-Experiment: λ=(0.3, 0, 0) wie CS 1, aber pro "
        "Entanglement zufällig auf die Positionen permutiert "
        "(PERMUTED_CONSTANTS). Ziel: bessere Generalisierung auf zufällig "
        "verteilte Fehler ohne volle Domain-Randomization.",
        constants=make_study_constants(
            lambdas=(0.3, 0.0, 0.0),
            lambda_strategy=LambdaStrategy.PERMUTED_CONSTANTS,
        ),
        hyperparams=STANDARD_HYPERPARAMS,
        coherence_times=CT_ALL,
    ),
}
