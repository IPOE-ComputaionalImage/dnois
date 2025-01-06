from numbers import Real

import torch

from ..base.typing import Ts, Sequence

__all__ = [
    'positive',
]


def positive(v: Real | Sequence[Real] | Ts, name: str = None):
    identifier = '' if name is None else ': ' + name

    if torch.is_tensor(v):
        if v.is_complex():
            raise TypeError('Cannot check the sign of a complex tensor' + identifier)
        if v.le(0).any():
            raise ValueError(f'Expected all positive' + identifier)
    elif isinstance(v, Sequence):
        if any(i <= 0 for i in v):
            raise ValueError(f'Expected all positive' + identifier + ', got ' + str(v))
    elif v <= 0:
        raise ValueError(f'Expected positive' + identifier + ', got ' + str(v))
