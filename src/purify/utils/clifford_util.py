from purify.entanglement import Entanglement


class Cliford:

    #will return a new Entanglement in the future

    def prot_1_jump_function(e_w: Entanglement, e_bd: Entanglement):
        oben = (
            4 * e_bd.get_current_lambda_1()
            + 3 * e_bd.get_current_lambda_2()
            + 3 * e_bd.get_current_lambda_3()
            - 3
        ) * e_w.get_current_fidelity() - e_bd.get_current_lambda_1()

        unten = (
            (4 * e_bd.get_current_lambda_2() + 4 * e_bd.get_current_lambda_3() - 2)
            * e_w.get_current_fidelity()
            - e_bd.get_current_lambda_2()
            - e_bd.get_current_lambda_3()
            - 1
        )
        return oben / unten

    def prot_1_success_probability(e_w: Entanglement, e_bd: Entanglement):
        return (2 / 3) * (
            1 - 2 * e_bd.get_current_lambda_2() - 2 * e_bd.get_current_lambda_3()
        ) * e_w.get_current_fidelity() + (1 / 3) * (
            1 + e_bd.get_current_lambda_2() + e_bd.get_current_lambda_3()
        )

    def prot_2_jump_function(e_w: Entanglement, e_bd: Entanglement):
        oben = (
            3 * e_bd.get_current_lambda_1()
            + 4 * e_bd.get_current_lambda_2()
            + 3 * e_bd.get_current_lambda_3()
            - 3
        ) * e_w.get_current_fidelity() - e_bd.get_current_lambda_2()
        unten = (
            (4 * e_bd.get_current_lambda_1() + 4 * e_bd.get_current_lambda_3() - 2)
            * e_w.get_current_fidelity()
            - e_bd.get_current_lambda_1()
            - e_bd.get_current_lambda_3()
            - 1
        )
        return oben / unten

    def prot_2_success_probability(e_w: Entanglement, e_bd: Entanglement):
        return (2 / 3) * (
            1 - 2 * e_bd.get_current_lambda_3() - 2 * e_bd.get_current_lambda_1()
        ) * e_w.get_current_fidelity() + (1 / 3) * (
            1 + e_bd.get_current_lambda_3() + e_bd.get_current_lambda_1()
        )

    def prot_3_jump_function(e_w: Entanglement, e_bd: Entanglement):
        oben = (
            3 * e_bd.get_current_lambda_1()
            + 3 * e_bd.get_current_lambda_2()
            + 4 * e_bd.get_current_lambda_3()
            - 3
        ) * e_w.get_current_fidelity() - e_bd.get_current_lambda_3()
        unten = (
            (4 * e_bd.get_current_lambda_1() + 4 * e_bd.get_current_lambda_2() - 2)
            * e_w.get_current_fidelity()
            - e_bd.get_current_lambda_1()
            - e_bd.get_current_lambda_2()
            - 1
        )
        return oben / unten

    def prot_3_success_probability(e_w: Entanglement, e_bd: Entanglement):
        return (2 / 3) * (
            1 - 2 * e_bd.get_current_lambda_1() - 2 * e_bd.get_current_lambda_2()
        ) * e_w.get_current_fidelity() + (1 / 3) * (
            1 + e_bd.get_current_lambda_1() + e_bd.get_current_lambda_2()
        )
