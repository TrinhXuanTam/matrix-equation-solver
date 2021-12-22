import numpy as np

from src.utils.math import is_stopping_criterion_satisfied


def iterative_method(
        A: np.ndarray,
        b: np.ndarray,
        U: np.ndarray,
        v: np.ndarray,
        x_0: np.ndarray,
) -> tuple[np.ndarray, int]:
    iterations_cnt = 0
    x_k = x_0

    while not is_stopping_criterion_satisfied(A, b, x_k):
        x_k = v + U @ x_k
        iterations_cnt = iterations_cnt + 1

    return x_k, iterations_cnt


