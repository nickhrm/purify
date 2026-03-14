from dataclasses import dataclass

from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy
from ppo.hyperparams_tuple import HyperparamsTuple


@dataclass
class CaseStudy:
    """
    Definiert eine Case Study: eine festgelegte Umgebungskonfiguration (Constants)
    und eine Liste zu trainierender Kohärenzzeiten.

    Jedes Gerät kann eine Teilmenge der `coherence_times` trainieren.
    Die eigentliche coherence_time wird erst beim Trainingsstart per
    `make_constants()` in das ConstantsTuple eingesetzt.
    """

    id: int
    constants: ConstantsTuple
    hyperparams: HyperparamsTuple
    coherence_times: list[float]

    def make_constants(self, coherence_time: float) -> ConstantsTuple:
        """Gibt ein neues ConstantsTuple zurück, bei dem coherence_time gesetzt ist."""
        return self.constants._replace(coherence_time=coherence_time)

    def folder_name(self) -> str:
        return f"case_study_{self.id}"

    @staticmethod
    def coherence_time_str(coherence_time: float) -> str:
        """Erzeugt einen dateisystemfreundlichen Ordnernamen für eine Kohärenzzeit."""
        return f"T{str(coherence_time).replace('.', '_')}"


# ─── Case Studies ─────────────────────────────────────────────────────────────
# Jede Case Study hat eine eindeutige ID (1, 2, 3, …).
# Die coherence_time wird als Platzhalter auf 0.0 gesetzt;
# der tatsächliche Wert wird von make_constants() injiziert.
# ─────────────────────────────────────────────────────────────────────────────

CASE_STUDIES: dict[int, CaseStudy] = {
    1: CaseStudy(
        id=1,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
    ),
    2: CaseStudy(
        id=2,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.0, 0.3, 0.0),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
    ),
    3: CaseStudy(
        id=3,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.0, 0.0, 0.3),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
    ),
    4: CaseStudy(
        id=4,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.RANDOM,
            lambdas=(0.0, 0.0, 0.0),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
    ),
    5: CaseStudy(
        id=5,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.3, 0.0, 0.0),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
                Action.PMD,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
        ],
    ),
    6: CaseStudy(
        id=6,
        constants=ConstantsTuple(
            coherence_time=0.0,  # Platzhalter – wird per make_constants() überschrieben
            pumping_probability=1,
            waiting_time_sensitivity=1,
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            lambdas=(0.2, 0.1, 0.0),
            actions=(
                Action.REPLACE,
                Action.PROT_1,
                Action.PROT_2,
                Action.PROT_3,
            ),
            min_fidelity=0.7,
            max_fidelity=0.7,
        ),
        hyperparams=HyperparamsTuple(
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
        ),
        coherence_times=[
            0.004,
            0.005,
            0.006,
            0.007,
            0.008,
            0.009,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.1,
        ],
    ),
}
