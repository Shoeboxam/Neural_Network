import numpy as np

__all__ = ['Variable']
_scalar = [str, int, float]


class Variable(np.ndarray):
    """Datatype for differentiable variables"""
    # Custom operations for 3D and certain non-conformable arrays
    # Enforces mutability for all numerics

    def __new__(cls, a):
        obj = np.array(a).view(cls)
        return obj

    def __add__(self, other):
        """Implicitly broadcast lesser operand to a higher conformable dimension"""
        if type(self) in _scalar or type(other) in _scalar:
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
        # Prevents broken reference
        if self.ndim == 2 and other.ndim == 2:
            return super().__iadd__(other)

        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __isub__(self, other):
        return self.__add__(-other)

    def __matmul__(self, other):
        """Implicitly broadcast and vectorize matrix multiplication along axis 3"""
        if type(self) in _scalar or type(other) in _scalar:
            return self * other

        # Stimuli id represents dimension 3.
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if self.ndim > 2 or other.ndim > 2:
            return Array(np.einsum('ij...,jl...->il...', self, other))
        return super().__matmul__(other)

    @property
    def T(self):
        return Array(np.swapaxes(self, 0, 1))