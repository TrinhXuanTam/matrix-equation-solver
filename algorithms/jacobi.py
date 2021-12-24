import numpy as np

from algorithms.iterative_method import iterative_method
from utils.math import spectral_radius, is_strictly_diagonally_dominant


def jacobi(A: np.ndarray, b: np.ndarray, x_0: np.ndarray) -> None:
    D = np.diag(np.diag(A))
    Q = D

    # Check convergence criterion
    is_spectral_radius_satisfied, \
    is_equation_array_strictly_diagonally_dominant, \
    is_iteration_matrix_norm_satisfied = check_convergence_criterion(A, Q)

    if not is_spectral_radius_satisfied:
        print_results()
        return

    solution, iterations_cnt = iterative_method(A, b, Q, x_0)

    print_results(
        is_spectral_radius_satisfied,
        is_equation_array_strictly_diagonally_dominant,
        is_iteration_matrix_norm_satisfied,
        iterations_cnt,
    )


def check_convergence_criterion(A: np.ndarray, Q: np.ndarray) -> tuple[bool, bool, bool]:
    is_spectral_radius_satisfied = spectral_radius(A, Q) < 1
    is_equation_array_strictly_diagonally_dominant = is_strictly_diagonally_dominant(A)
    is_iteration_matrix_norm_satisfied = np.linalg.norm(np.linalg.solve(Q, Q - A), ord=2) < 1

    return is_spectral_radius_satisfied, is_equation_array_strictly_diagonally_dominant, is_iteration_matrix_norm_satisfied


def print_results(
        is_spectral_radius_satisfied=False,
        is_equation_array_strictly_diagonally_dominant=False,
        is_iteration_matrix_norm_satisfied=False,
        iterations_cnt=0,
) -> None:
    print("JACOBI METHOD")
    print(f"\tspectral radius: {is_spectral_radius_satisfied}")
    print(f"\tstrictly diagonally dominant: {is_equation_array_strictly_diagonally_dominant}")
    print(f"\titeration matrix norm: {is_iteration_matrix_norm_satisfied}")
    print(f"\titerations: {iterations_cnt}")
