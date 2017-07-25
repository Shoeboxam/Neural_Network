import numpy as np


# Define custom operations for adding and multiplying 3D and mismatched arrays
class Array(np.ndarray):
    _types = [str, int]

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    def __add__(self, other):
        """Implicitly broadcast to a higher dimension"""
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

    def __matmul__(self, other):
        """Implicitly broadcast and vectorize matrix multiplication"""
        if type(self) in self._types or type(other) in self._types:
            return self * other

        # Stimuli id represents dimension 3.
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if self.ndim > 2 or other.ndim > 2:
            return Array(np.einsum('ij...,jl...->il...', self, other))
        return super().__matmul__(other)
