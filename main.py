import numpy as np

from algorithms.gauss_seidel import gauss_seidel
from algorithms.jacobi import jacobi

if __name__ == '__main__':
    gammas = [5, 2, 1.2]
    x_0 = np.zeros(shape=(20, 1))

    for gamma in gammas:
        print("--------------------------")
        print(f"GAMMA: {gamma}")
        print()

        # Generate equation matrix from task assignment
        A = np.zeros(shape=(20, 20))
        np.fill_diagonal(A, gamma)
        indices = np.arange(19)
        A[indices, indices + 1] = A[indices + 1, indices] = -1

        # Generate right side vector from task assignment
        b = np.full(shape=(20, 1), fill_value=gamma - 2)
        b[0] = b[len(b) - 1] = gamma - 1

        # Solve with Jacobi method
        jacobi(A, b, x_0)

        print()

        # Solve with Gauss-Seidel method
        gauss_seidel(A, b, x_0)

        print("--------------------------")
        print()
