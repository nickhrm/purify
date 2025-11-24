
import math


def decibel_to_linear(dec_val):
    return math.pow(10, dec_val / 10)


def linear_to_decibel(db_val):
    if db_val <= 0:
        return -math.inf

    return 10 * math.log10(db_val)

