import numpy as np

from src.algorithms.jacobi import jacobi

if __name__ == '__main__':
    gammas = [5, 2, 1.2]
    initial_guess = np.zeros(shape=(20, 1))

    for gamma in gammas:
        # Generate equation matrix
        equation_matrix = np.zeros(shape=(20, 20))
        np.fill_diagonal(equation_matrix, gamma)
        indices = np.arange(19)
        equation_matrix[indices, indices + 1] = equation_matrix[indices + 1, indices] = -1

        # Generate right side vector
        right_side_vector = np.full(shape=(20, 1), fill_value=gamma - 2)
        right_side_vector[0] = right_side_vector[len(right_side_vector) - 1] = gamma - 1

        # Solve with Jacobi method
        jacobi(equation_matrix, right_side_vector, initial_guess)
