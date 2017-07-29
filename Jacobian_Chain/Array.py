import numpy as np


class Array(np.ndarray):
    """Custom operations for 3D and certain non-conformable arrays"""
    _types = [str, int, float]

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    def __add__(self, other):
        """Implicitly broadcast lesser operand to a higher conformable dimension"""
        if type(self) in self._types or type(other) in self._types:
            return super().__add__(other)

        # Stimuli become vectorized, but bias units remain 1D. To add wx + b, must broadcast
        if self.ndim == 2 and other.ndim == 1:
            return Array(np.add(self, np.tile(other[..., np.newaxis], self.shape[1])))
        if self.ndim == 1 and other.ndim == 2:
            return Array(np.add(np.tile(self[..., np.newaxis], other.shape[1]), other))

        if self.ndim == 3 and other.ndim == 2:
            return Array(np.add(self, np.tile(other[..., np.newaxis], self.shape[2])))
        if self.ndim == 2 and other.ndim == 3:
            return Array(np.add(np.tile(self[..., np.newaxis], other.shape[2]), other))
        return np.add(self, other)

    def __iadd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __isub__(self, other):
        return self.__add__(-other)

    def __truediv__(self, other):
        if type(self) in self._types or type(other) in self._types:
            return super().__truediv__(other)

        # Adagrad has a 3D elementwise division
        if self.ndim == 2 and other.ndim == 1:
            return Array(np.divide(self, np.tile(other[..., np.newaxis], self.shape[1])))
        if self.ndim == 1 and other.ndim == 2:
            return Array(np.divide(np.tile(self[..., np.newaxis], other.shape[1]), other))

        if self.ndim == 3 and other.ndim == 2:
            return Array(np.divide(self, np.tile(other[..., np.newaxis], self.shape[2])))
        if self.ndim == 2 and other.ndim == 3:
            return Array(np.divide(np.tile(self[..., np.newaxis], other.shape[2]), other))
        return np.divide(self, other)

    def __matmul__(self, other):
        """Implicitly broadcast and vectorize matrix multiplication along axis 3"""
        if type(self) in self._types or type(other) in self._types:
            return self * other

        # Stimuli id represents dimension 3.
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if self.ndim > 2 or other.ndim > 2:
            return Array(np.einsum('ij...,jl...->il...', self, other))
        return super().__matmul__(other)

    @property
    def T(self):
        return Array(np.swapaxes(self, 0, 1))


# Sanity check to ensure 3D matmul is correct
# from .Function import diag

# def check_mult(A, B):
#     correct = True
#     product = A @ B
#
#     if A.ndim == 3 and B.ndim == 3:
#         for idx in range(np.shape(product)[2]):
#             if not np.allclose(A[:, :, idx] @ B[:, :, idx], product[..., idx]):
#                 correct = False
#
#     elif A.ndim == 2 and B.ndim == 3:
#         for idx in range(np.shape(product)[2]):
#             if not np.allclose(A @ B[:, :, idx], product[..., idx]):
#                 correct = False
#
#     elif A.ndim == 3 and B.ndim == 2:
#         for idx in range(np.shape(product)[2]):
#             if not np.allclose(A[:, :, idx] @ B, product[..., idx]):
#                 correct = False
#
#     else:
#         return False
#
#     return correct


# A = np.random.rand(4, 2).view(Array)
# B = np.random.rand(4).view(Array)

# print(np.shape(A @ B))
# print(check_mult(A, B))
