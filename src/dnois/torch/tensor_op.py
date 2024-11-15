from ..base.typing import Ts

__all__ = [
    'as1d',
]


def as1d(x: Ts, ndim: int = 1, dim: int = -1) -> Ts:
    shape = [1 for _ in range(ndim)]
    shape[dim] = -1
    return x.reshape(shape)
