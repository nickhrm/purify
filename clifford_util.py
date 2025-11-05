

class Cliford:
    def prot_1_jump_function(f: float, f_bd: float):
        return ((3 * (1 - f_bd) - 3) * f) / ((4 * (1 - f_bd) - 2) * f - 1)

    def prot_1_success_probability(f: float, f_bd: float):
        return (2 / 3) * (1 - 2 * (1 - f_bd)) * f + (1 / 3) * (1 + (1 - f_bd))

    def prot_2_jump_function(f: float, f_bd: float):
        return ((3 * (1 - f_bd) - 3) * f) / ((4 * (1 - f_bd) - 2) * f - 1)

    def prot_2_success_probability(f: float, f_bd: float):
        return (2 / 3) * (1 - 2 * (1 - f_bd)) * f + (1 / 3) * (1 + (1 - f_bd))

    def prot_3_jump_function(f: float, f_bd: float):
        return ((4 * (1 - f_bd) - 3) * f - (1 - f_bd)) / (2 * f - 1)

    def prot_3_success_probability(f: float, f_bd: float):
        return max(0, min((2 / 3) * f + (1 / 3),1))
