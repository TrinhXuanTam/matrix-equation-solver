import numpy as np


def spectral_radius(matrix: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvals(matrix)
    print(eigenvalues)
    return max(eigenvalues)


def is_stopping_criterion_satisfied(
        equation_matrix: np.ndarray,
        right_side_vector: np.ndarray,
        solution: np.ndarray
) -> bool:
    precision = np.linalg.norm(equation_matrix @ solution - right_side_vector) / np.linalg.norm(right_side_vector)
    return precision < 10e-6


def is_spectral_radius_condition_satisfied(equation_matrix: np.ndarray, regular_matrix: np.ndarray) -> bool:
    identity_matrix = np.identity(n=regular_matrix.shape[0])
    eigenvalues = np.linalg.eigvals(identity_matrix - np.linalg.solve(regular_matrix, equation_matrix))
    return np.max(np.abs(eigenvalues)) < 1
