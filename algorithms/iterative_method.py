import numpy as np

from utils.math import is_stopping_criterion_satisfied


def iterative_method(
        A: np.ndarray,
        b: np.ndarray,
        U: np.ndarray,
        v: np.ndarray,
        x_0: np.ndarray,
) -> tuple[np.ndarray, int]:
    iterations_cnt = 0
    x = x_0

    # Iterate until stopping criterion is not satisfied
    while not is_stopping_criterion_satisfied(A, b, x):
        # Calculate next solution x where U represents iteration matrix and v vector
        x = v + U @ x
        iterations_cnt = iterations_cnt + 1

    # Return result and number of iterations
    return x, iterations_cnt
