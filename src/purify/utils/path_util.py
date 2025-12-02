


from purify.my_constants import LAMBDA_1, LAMBDA_2, LAMBDA_3


def path_from_lambdas():
    return f"ALL_RESULTS_{str(LAMBDA_1).replace(".", "")}_{str(LAMBDA_2).replace(".", "")}_{str(LAMBDA_3).replace(".", "")}.csv"