import numpy as np


def is_stopping_criterion_satisfied(
        equation_matrix: np.ndarray,
        right_side_vector: np.ndarray,
        solution: np.ndarray
) -> bool:
    precision = np.linalg.norm(equation_matrix @ solution - right_side_vector, ord=2) / np.linalg.norm(
        right_side_vector, ord=2)
    return precision < 10e-6


def spectral_radius(equation_matrix: np.ndarray, regular_matrix: np.ndarray) -> float:
    identity_matrix = np.identity(n=regular_matrix.shape[0])
    eigenvalues = np.linalg.eigvals(identity_matrix - np.linalg.solve(regular_matrix, equation_matrix))
    return np.max(np.abs(eigenvalues))


def is_symmetric(matrix: np.ndarray) -> bool:
    return (matrix == matrix.T).all()


def is_positively_definite(matrix):
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_strictly_diagonally_dominant(matrix: np.ndarray) -> bool:
    diagonal_coefficient = np.diag(np.abs(matrix))
    row_sum_without_diagonal = np.sum(np.abs(matrix), axis=1) - diagonal_coefficient
    return np.all(diagonal_coefficient > row_sum_without_diagonal)
