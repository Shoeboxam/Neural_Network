import numpy as np


# Define custom operations for adding and multiplying 3D and mismatched arrays
class Array(np.ndarray):

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    def __add__(self, other):
        """Implicitly broadcast to a higher dimension"""

        # Stimuli become vectorized, but bias units remain 1D. To add wx + b, must broadcast
        if self.ndim == 2 and other.ndim == 1:
            return np.add(self, np.tile(other[..., np.newaxis], np.shape(self)[1]))
        elif self.ndim == 1 and other.ndim == 2:
            return np.add(np.tile(self[..., np.newaxis], np.shape(other)[1]), other)
        else:
            return np.add(self, other)

    def __matmul__(self, other):
        """Implicitly broadcast and vectorize matrix multiplication"""

        # Stimuli id represents dimension 3.
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if self.ndim == 3 and other.ndim == 3:
            return np.einsum('ijn,jln->iln', self, other)
        if self.ndim == 2 and other.ndim == 3:
            return np.einsum('ij,jln->iln', self, other)
        if self.ndim == 3 and other.ndim == 2:
            return np.einsum('ijn,jl->iln', self, other)
        return super().__matmul__(other)


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

print(np.shape(A @ B))

# print(np.shape(Array(np.zeros(4))))


def diag(array):
    if array.ndim == 1:
        return np.diag(array)
    else:
        elements = []
        for idx in range(array.shape[-1]):
            elements.append(diag(array[..., idx]))
        return np.stack(elements, array.ndim)

print(diag(np.random.rand(4, 30)).shape)
