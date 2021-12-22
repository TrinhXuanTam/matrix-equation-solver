import numpy as np


def is_stopping_criterion_satisfied(
        equation_matrix: np.ndarray,
        right_side_vector: np.ndarray,
        solution: np.ndarray
) -> bool:
    # Check if stopping criterion from task assignment is satisfied
    precision = np.linalg.norm(equation_matrix @ solution - right_side_vector, ord=2) / np.linalg.norm(right_side_vector, ord=2)
    return precision < 10e-6


def spectral_radius(A: np.ndarray, Q: np.ndarray) -> float:
    # Find the eigenvalues of matrix: E - Q^-1 * A
    E = np.identity(n=Q.shape[0])
    eigenvalues = np.linalg.eigvals(E - np.linalg.solve(Q, A))

    # Return the largest absolute value from eigenvalues
    return np.max(np.abs(eigenvalues))


def is_symmetric(matrix: np.ndarray) -> bool:
    # Symmetric matrix is a square matrix that is equal to its transpose
    return (matrix == matrix.T).all()


def is_positively_definite(matrix):
    # A matrix is positive definite if it's symmetric and all its eigenvalues are positive
    return np.all(np.linalg.eigvals(matrix) > 0)


def is_strictly_diagonally_dominant(matrix: np.ndarray) -> bool:
    # Create array of absolute values of diagonal elements
    diagonal_coefficient = np.diag(np.abs(matrix))

    # Create array from sum of rows without diagonal elements
    row_sum_without_diagonal = np.sum(np.abs(matrix), axis=1) - diagonal_coefficient

    # A square matrix is strictly diagonally dominant if, for every row of the matrix,
    # the absolute value of the diagonal entry in a row is larger than the sum of all
    # the other entries in that row without the diagonal entry
    return np.all(diagonal_coefficient > row_sum_without_diagonal)
