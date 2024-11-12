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

from .exception import ShapeError

__all__ = [
    'is_scalar',
    'is_tensor',
    'pair',
    'scalar',
    'scl_or_vec',
    'size2d',
    'sizend',
    'vector',

    'ConvOut',
    'FovSeg',
    'Numeric',
    'PsfCenter',
    'RGBFormat',
    'Scalar',
    'SclOrVec',
    'Size2d',
    'Sizend',
    'Spacing',
    'Tensor',
    'Ts',
    'Vector',
]
__all__ += typing.__all__

_T = TypeVar('_T')

_dty = torch.dtype
_dev = torch.device
Device = Union[str, int, _dev]  # as same as torch.DeviceLikeType

# tensor-like
Ts = Tensor
Spacing = Union[float, Ts]  # delta (grid spacing) type
Numeric = Union[float, Ts]  # support numeric operation
Scalar = Union[float, Ts]  # can be converted to 0d tensor
Vector = Union[float, Sequence[float], Ts]  # can be converted to 1d tensor
SclOrVec = Union[float, Sequence[float], Ts]  # Scalar or Vector

Size2d = Union[int, tuple[int, int]]
Sizend = Union[int, Sequence[int]]

# options
FovSeg = Literal['paraxial', 'pointwise']
ConvOut = Literal['full', 'same', 'valid']
RGBFormat = Literal['floats', 'ints', 'hex']
PsfCenter = Literal['linear', 'mean', 'chief']


def size2d(size: Size2d) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    elif isinstance(size, tuple):
        if not isinstance(size[0], int) or not isinstance(size[1], int):
            raise ValueError(f'A pair of int expected, got {size}')
        # allow negative
        return cast(tuple[int, int], size)
    else:
        raise ValueError(f'An int or a pair of int expected, got {type(size)}')


def sizend(size: Sizend, ndim: int = None) -> list[int]:
    if isinstance(size, int):
        return [size for _ in range(ndim)]
    elif isinstance(size, Sequence):
        if not all(isinstance(s, int) for s in size):
            raise ValueError(f'A sequence of int expected, got {size}')
        if ndim is not None and len(size) != ndim:
            raise ValueError(f'{ndim} integers expected, got {len(size)}')
        return list(size)
    else:
        raise ValueError(f'An int or a sequence of int expected, got {type(size)}')


def vector(arg: Vector, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    cfg = {}
    if dtype is not None:
        cfg['dtype'] = dtype
    if device is not None:
        cfg['device'] = device
    if isinstance(arg, float):
        return torch.tensor([arg], **cfg)
    elif isinstance(arg, Sequence) and all(isinstance(item, float) for item in arg):
        return torch.tensor(arg, **cfg)
    elif is_tensor(arg):
        if arg.ndim == 0:
            arg = arg.unsqueeze(0)
        if arg.ndim != 1:
            raise ShapeError(f'Trying to convert a tensor with shape {arg.shape} to a vector')
        return arg.to(**cfg, **kwargs)
    else:
        raise TypeError(f'A float, a sequence of float or a 1d tensor expected, got {type(arg)}')


def scalar(arg: Scalar, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    cfg = {}
    if dtype is not None:
        cfg['dtype'] = dtype
    if device is not None:
        cfg['device'] = device
    if isinstance(arg, float):
        return torch.tensor(arg, **cfg)
    elif is_tensor(arg):
        if arg.ndim != 0:
            raise ShapeError(f'Trying to convert a tensor with shape {arg.shape} to a scalar')
        return arg.to(**cfg, **kwargs)
    else:
        raise TypeError(f'A float or a 0d tensor is expected, got {type(arg)}')


def is_scalar(arg: Any) -> bool:
    return isinstance(arg, float) or (is_tensor(arg) and arg.ndim == 0)


def scl_or_vec(arg: SclOrVec, dtype: _dty = None, device: Device = None, **kwargs) -> Ts:
    cfg = {}
    if dtype is not None:
        cfg['dtype'] = dtype
    if device is not None:
        cfg['device'] = device
    if isinstance(arg, float):
        return torch.tensor(arg, **cfg)
    elif isinstance(arg, Sequence) and all(isinstance(item, float) for item in arg):
        return torch.tensor(arg, **cfg)
    elif is_tensor(arg):
        if arg.ndim > 1:
            raise ShapeError(f'Trying to convert a tensor with shape {arg.shape} to a scalar or vector')
        return arg.to(**cfg, **kwargs)
    else:
        raise TypeError(f'A float, a sequence of float or a 0d or 1d tensor expected, got {type(arg)}')


def pair(arg: Union[_T, tuple[_T, _T]]) -> tuple[_T, _T]:
    return arg if isinstance(arg, tuple) else (arg, arg)


def check_3d_vector(ts: Ts, name: str = 'a 3d vector'):
    if ts.ndim < 1 or ts.size(-1) != 3:
        raise ShapeError(f'Size of last dimension of {name} must be 3, got shape {ts.shape}')


def check_2d_vector(ts: Ts, name: str = 'a 2d vector'):
    if ts.ndim < 1 or ts.size(-1) != 2:
        raise ShapeError(f'Size of last dimension of {name} must be 2, got shape {ts.shape}')
