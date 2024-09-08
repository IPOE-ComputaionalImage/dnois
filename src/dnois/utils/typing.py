# The philosophy of this type system is, when multiple types for a variable
# (arguments, etc.) are allowed to improve flexibility and usability and
# there is a standard type for it so that all the allowed types can be
# converted to the standard type, a new type alias (typically constructed
# through Union) should be defined to cover allowed types and a converter
# function should be provided to convert them to the standard type.
# Converter functions must conduct rigorous type checking and should be named
# as the lower case of corresponding type alias.

import typing
from typing import *

import torch
from torch import is_tensor, Tensor

__all__ = [
    'is_scalar',
    'is_tensor',
    'pair',
    'param',
    'scalar',
    'size2d',

    'Spacing',
    'Numeric',
    'Param',
    'Scalar',
    'Size2d',
    'Tensor',
    'Ts',
]
__all__ += typing.__all__

_T = TypeVar('_T')

_dty = torch.dtype
_dev = torch.device
Ts = Tensor
Device = Union[str, int, _dev]  # as same as torch.DeviceLikeType

Spacing = Union[float, Ts]  # delta (grid spacing) type
Numeric = Union[int, float, Ts]  # support numeric operation
Scalar = Union[float, Ts]  # can be converted to 0d tensor
Param = Union[float, Sequence[float], Ts]  # can be converted to 1d tensor
Size2d = Union[int, tuple[int, int]]


def size2d(size: Size2d) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    elif isinstance(size, tuple):
        if len(size) != 2:
            raise ValueError(f'Too many elements as 2d size: {size}')
        if not isinstance(size[0], int) or not isinstance(size[1], int):
            raise ValueError(f'A pair of int expected, got {size}')
        # allow negative
        return cast(tuple[int, int], size)
    else:
        raise ValueError(f'An int or a pair of int expected, got {type(size)}')


def param(arg: Param, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    if isinstance(arg, float):
        return torch.tensor([arg], dtype, device)
    elif isinstance(arg, Sequence) and all(isinstance(item, float) for item in arg):
        return torch.tensor(arg, dtype, device)
    elif is_tensor(arg):
        if arg.ndim == 0:
            arg = arg.unsqueeze(0)
        if arg.ndim != 1:
            raise ValueError(f'System parameters must be a scalar or 1d tensor. Got {arg.shape}')
        return arg.to(device, dtype, **kwargs)
    else:
        raise TypeError(f'A float, a sequence of float or a 1d tensor is expected, got {type(arg)}')


def scalar(arg: Scalar, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    if isinstance(arg, float):
        return torch.tensor(arg, dtype, device)
    elif is_tensor(arg):
        if arg.ndim != 0:
            raise ValueError(f'Trying to convert a tensor with shape ({arg.shape}) to a scalar')
        return arg.to(device, dtype, **kwargs)
    else:
        raise TypeError(f'A float or a 0d tensor is expected, got {type(arg)}')


def is_scalar(arg: Any) -> bool:
    return isinstance(arg, float) or (is_tensor(arg) and arg.ndim == 0)


def pair[_T](arg: Union[_T, tuple[_T, _T]]) -> tuple[_T, _T]:
    return arg if isinstance(arg, tuple) else (arg, arg)
