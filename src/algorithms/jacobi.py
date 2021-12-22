import numpy as np

from src.utils.math import is_stopping_criterion_satisfied, is_spectral_radius_condition_satisfied


def jacobi(equation_matrix: np.ndarray, right_side_vector: np.ndarray, initial_guess: np.ndarray) -> None:
    diagonal_matrix = np.diag(np.diag(equation_matrix))
    remaining_matrix = equation_matrix - diagonal_matrix
    solution = initial_guess
    iterations_cnt = 0

    if not is_spectral_radius_condition_satisfied(equation_matrix, diagonal_matrix):
        return

    dinv_b = np.linalg.solve(diagonal_matrix, right_side_vector)
    while not is_stopping_criterion_satisfied(equation_matrix, right_side_vector, solution):
        solution = dinv_b - np.linalg.solve(diagonal_matrix, remaining_matrix @ solution)
        iterations_cnt = iterations_cnt + 1
