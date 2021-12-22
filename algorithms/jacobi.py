import numpy as np

from algorithms.iterative_method import iterative_method
from utils.math import spectral_radius, is_strictly_diagonally_dominant


def jacobi(A: np.ndarray, b: np.ndarray, x_0: np.ndarray) -> None:
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)
    Q = D

    # Calculate iteration matrix U_j and vector v_j
    U_j = np.linalg.solve(Q, -U - L)
    v_j = np.linalg.solve(Q, b)

    # Check convergence criterion
    is_spectral_radius_satisfied = spectral_radius(A, D) < 1
    is_equation_array_strictly_diagonally_dominant = is_strictly_diagonally_dominant(A)
    is_iteration_matrix_norm_satisfied = np.linalg.norm(U_j, ord=2) < 1

    if not is_spectral_radius_satisfied:
        print_results()
        return

    solution, iterations_cnt = iterative_method(A, b, U_j, v_j, x_0)

    print_results(
        is_spectral_radius_satisfied,
        is_equation_array_strictly_diagonally_dominant,
        is_iteration_matrix_norm_satisfied,
        iterations_cnt,
    )


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
