from typing import Tuple

import numpy as np

from utils.math import is_stopping_criterion_satisfied


def iterative_method(
        A: np.ndarray,
        b: np.ndarray,
        Q: np.ndarray,
        x_0: np.ndarray,
) -> Tuple[np.ndarray, int]:
    iterations_cnt = 0
    x = x_0

    # Calculate iteration matrix U = Q^-1 * (Q - A) and vector v = Q^-1 * b
    U = np.linalg.solve(Q, Q - A)
    v = np.linalg.solve(Q, b)

    # Iterate until stopping criterion is not satisfied
    while not is_stopping_criterion_satisfied(A, b, x):
        # Calculate next solution x_(k+1) = v + U * x_k
        x = v + U @ x

        # Increment iteration counter
        iterations_cnt = iterations_cnt + 1

    # Return result and number of iterations
    return x, iterations_cnt
