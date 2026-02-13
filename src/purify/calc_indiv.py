from purify.my_enums import Action
from purify.entanglement import Entanglement
from purify.my_time import Time
from purify.utils.purification_util import Purification


def test_pur():
    time = Time()

    coherence_time = 0.05
    action = Action.PROT_2

    good_mem = Entanglement(
        time,
        time.get_current_time(),
        creation_fidelity=0.6,
        creation_lambda_1=0.1,
        creation_lambda_2=0.1,
        creation_lambda_3=0.1,
        decoherence_time=coherence_time,
    )

    bad_mem = Entanglement(
        time,
        time.get_current_time(),
        creation_fidelity=0.7,
        creation_lambda_1=0.0,
        creation_lambda_2=0.3,
        creation_lambda_3=0.0,
        decoherence_time=coherence_time,
    )

    resulting_fidelity =  Purification.jump_function_from_action(good_mem, bad_mem, action)
    success_prob = Purification.success_probability_from_action(good_mem, bad_mem, action)


    print(f"action {action} resulting f: {resulting_fidelity}, success_prob: {success_prob}")



if __name__ == "__main__":
    test_pur()
