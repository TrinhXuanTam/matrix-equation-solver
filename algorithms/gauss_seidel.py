import numpy as np

from algorithms.iterative_method import iterative_method
from utils.math import spectral_radius, is_strictly_diagonally_dominant, is_symmetric, is_positively_definite


def gauss_seidel(A: np.ndarray, b: np.ndarray, x_0: np.ndarray) -> None:
    D = np.diag(np.diag(A))
    U = np.triu(A, k=1)
    L = np.tril(A, k=-1)
    Q = D + L

    # Calculate iteration matrix U_g and vector v_g
    U_g = np.linalg.solve(Q, -U)
    v_g = np.linalg.solve(Q, b)

    # Check convergence criterion
    is_spectral_radius_satisfied = spectral_radius(A, D) < 1
    is_equation_array_strictly_diagonally_dominant = is_strictly_diagonally_dominant(A)
    is_iteration_matrix_norm_satisfied = np.linalg.norm(U_g, ord=2) < 1
    is_symmetric_and_positively_definite = is_symmetric(A) and is_positively_definite(A)

    if not is_spectral_radius_satisfied:
        print_results()
        return

    solution, iterations_cnt = iterative_method(A, b, U_g, v_g, x_0)

    print_results(
        is_spectral_radius_satisfied,
        is_equation_array_strictly_diagonally_dominant,
        is_iteration_matrix_norm_satisfied,
        is_symmetric_and_positively_definite,
        iterations_cnt,
    )


def print_results(
        is_spectral_radius_satisfied=False,
        is_equation_array_strictly_diagonally_dominant=False,
        is_iteration_matrix_norm_satisfied=False,
        is_symmetric_and_positively_definite=False,
        iterations_cnt=0,
) -> None:
    print("GAUSS-SEIDEL METHOD")
    print(f"\tspectral radius: {is_spectral_radius_satisfied}")
    print(f"\tstrictly diagonally dominant: {is_equation_array_strictly_diagonally_dominant}")
    print(f"\titeration matrix norm: {is_iteration_matrix_norm_satisfied}")
    print(f"\tsymmetric and positively definite: {is_symmetric_and_positively_definite}")
    print(f"\titerations: {iterations_cnt}")
