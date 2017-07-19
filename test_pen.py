import numpy as np

from Jacobian_Chain.Function import diag
from Jacobian_Chain.Neural_Network import Array

A = np.random.rand(4, 2, 7).view(Array)
B = np.random.rand(2, 5, 7).view(Array)

# print(Array([1, 2]))
# print(A.ndim)


def check_mult(A, B):
    correct = True
    product = A @ B

    if A.ndim == 3 and B.ndim == 3:
        for idx in range(np.shape(product)[2]):
            if not np.allclose(A[:, :, idx] @ B[:, :, idx], product[..., idx]):
                correct = False

    elif A.ndim == 2 and B.ndim == 3:
        for idx in range(np.shape(product)[2]):
            if not np.allclose(A @ B[:, :, idx], product[..., idx]):
                correct = False

    elif A.ndim == 3 and B.ndim == 2:
        for idx in range(np.shape(product)[2]):
            if not np.allclose(A[:, :, idx] @ B, product[..., idx]):
                correct = False

    else:
        return False

    return correct

# print(check_mult(A, B))


#
# A = np.random.rand(4, 2).view(Array)
# B = np.random.rand(4).view(Array)

# print(np.shape(A @ B))

# print(np.shape(Array(np.zeros(4))))

A = np.random.rand(4, 30)


def check_diag(A):
    correct = True
    A_diagon = diag(A)

    for rowid in range(A.shape[1]):
        if not np.allclose(np.diag(A.T[rowid]), A_diagon[..., rowid]):
            correct = False

    return correct
print(check_diag(A))


