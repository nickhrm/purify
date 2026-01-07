import logging
from purify.entanglement import Entanglement
from purify.my_enums import Action


logger = logging.getLogger(__name__)


"""Assumes that e_good is in a Werner-State. As every Bell-Diagonal-State can be
transformed into a Werner-State using twirling, the lambdas can just be ignored"""


class Purification:

    @staticmethod
    def prot_1_jump_function(e_good: Entanglement, e_bad: Entanglement):
        oben = (
            4 * e_bad.get_current_lambda_1()
            + 3 * e_bad.get_current_lambda_2()
            + 3 * e_bad.get_current_lambda_3()
            - 3
        ) * e_good.get_current_fidelity() - e_bad.get_current_lambda_1()

        unten = (
            (4 * e_bad.get_current_lambda_2() + 4 * e_bad.get_current_lambda_3() - 2)
            * e_good.get_current_fidelity()
            - e_bad.get_current_lambda_2()
            - e_bad.get_current_lambda_3()
            - 1
        )
        return oben / unten

    @staticmethod
    def prot_1_success_probability(e_good: Entanglement, e_bad: Entanglement):
        base = (2 / 3) * (
            1 - 2 * e_bad.get_current_lambda_2() - 2 * e_bad.get_current_lambda_3()
        ) * e_good.get_current_fidelity() + (1 / 3) * (
            1 + e_bad.get_current_lambda_2() + e_bad.get_current_lambda_3()
        )
        logger.info("Success probability is %s", base)
        return base


    @staticmethod
    def prot_2_jump_function(e_good: Entanglement, e_bad: Entanglement):
        oben = (
            3 * e_bad.get_current_lambda_1()
            + 4 * e_bad.get_current_lambda_2()
            + 3 * e_bad.get_current_lambda_3()
            - 3
        ) * e_good.get_current_fidelity() - e_bad.get_current_lambda_2()
        unten = (
            (4 * e_bad.get_current_lambda_1() + 4 * e_bad.get_current_lambda_3() - 2)
            * e_good.get_current_fidelity()
            - e_bad.get_current_lambda_1()
            - e_bad.get_current_lambda_3()
            - 1
        )
        return oben / unten

    @staticmethod
    def prot_2_success_probability(e_good: Entanglement, e_bad: Entanglement):
        base = (2 / 3) * (
            1 - 2 * e_bad.get_current_lambda_3() - 2 * e_bad.get_current_lambda_1()
        ) * e_good.get_current_fidelity() + (1 / 3) * (
            1 + e_bad.get_current_lambda_3() + e_bad.get_current_lambda_1()
        )

        logger.info("Success probability is %s", base)
        return base

    @staticmethod
    def prot_3_jump_function(e_good: Entanglement, e_bad: Entanglement):
        oben = (
            3 * e_bad.get_current_lambda_1()
            + 3 * e_bad.get_current_lambda_2()
            + 4 * e_bad.get_current_lambda_3()
            - 3
        ) * e_good.get_current_fidelity() - e_bad.get_current_lambda_3()
        unten = (
            (4 * e_bad.get_current_lambda_1() + 4 * e_bad.get_current_lambda_2() - 2)
            * e_good.get_current_fidelity()
            - e_bad.get_current_lambda_1()
            - e_bad.get_current_lambda_2()
            - 1
        )
        return oben / unten

    @staticmethod
    def prot_3_success_probability(e_good: Entanglement, e_bad: Entanglement):
        base = (2 / 3) * (
            1 - 2 * e_bad.get_current_lambda_1() - 2 * e_bad.get_current_lambda_2()
        ) * e_good.get_current_fidelity() + (1 / 3) * (
            1 + e_bad.get_current_lambda_1() + e_bad.get_current_lambda_2()
        )

        logger.info("Success probability is %s", base)

        return base

    @staticmethod
    def pmd_jump_function(e_good: Entanglement, e_bad: Entanglement):
        if e_bad.get_current_lambda_2() != 0 or e_bad.get_current_lambda_3() != 0:
            raise Exception("pmd can only be used if lambda 2 and 3 are equal to 0")


        return (
            e_bad.get_current_fidelity()
            * e_good.get_current_fidelity()
            / Purification.pmd_success_probability(e_good, e_bad)
        )

    @staticmethod
    def pmd_success_probability(e_good: Entanglement, e_bad: Entanglement):
        if e_bad.get_current_lambda_2() != 0 or e_bad.get_current_lambda_3() != 0:
            raise Exception("pmd can only be used if lambda 2 and 3 are equal to 0")

        return e_bad.get_current_fidelity() * e_good.get_current_fidelity() + (
            1 - e_bad.get_current_fidelity()
        ) * (1 - e_good.get_current_fidelity())

    @staticmethod
    def success_probability_from_action(e_good: Entanglement, e_bad: Entanglement, action: Action):
        # 'self' wurde entfernt, da es eine @staticmethod ist
        match action:
            case Action.PROT_1:
                return Purification.prot_1_success_probability(e_good, e_bad)
            case Action.PROT_2:
                return Purification.prot_2_success_probability(e_good, e_bad)
            case Action.PROT_3:
                return Purification.prot_3_success_probability(e_good, e_bad)
            case Action.PMD:
                return Purification.pmd_success_probability(e_good, e_bad)
            case _:
                raise ValueError(f"Unknown action: {action}")


    @staticmethod
    def jump_function_from_action(e_good: Entanglement, e_bad: Entanglement, action: Action):
        """
        Gibt den neuen Fidelity-Wert (oder Parameter) basierend auf der gewählten Action zurück.
        Dies ist das Gegenstück zur Probability-Funktion.
        """
        match action:
            case Action.PROT_1:
                return Purification.prot_1_jump_function(e_good, e_bad)
            case Action.PROT_2:
                return Purification.prot_2_jump_function(e_good, e_bad)
            case Action.PROT_3:
                return Purification.prot_3_jump_function(e_good, e_bad)
            case Action.PMD:
                return Purification.pmd_jump_function(e_good, e_bad)
            case _:
                raise ValueError(f"Unknown action: {action}")
