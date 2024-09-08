import torch
from torch.nn import functional

from dnois.utils.typing import Ts, Spacing


def _check_dim(name: str, shape: torch.Size, dims: tuple[int, ...], **deltas: Spacing):
    ndim = len(shape)
    dims = [dim % ndim - ndim for dim in dims]
    for k, delta in deltas.items():
        if not torch.is_tensor(delta) or delta.ndim == 0:
            continue

        d_shape = delta.shape
        for _a, _b in zip(reversed(shape), reversed(d_shape)):
            if _a != _b and _a != 1 and _b != 1:
                raise ValueError(f'{name} ({shape}) and {k} ({d_shape}) cannot broadcast')
        for dim in dims:
            if dim >= -delta.ndim and delta.size(dim) != 1:
                raise ValueError(f'The dimension to be transformed of {k} ({d_shape}) is not 1.')


def _mul(x: Ts, *fs: Spacing) -> Ts:
    for f in fs:
        if f is not None:
            x = x * f
    return x


def _div(x: Ts, *fs: Spacing) -> Ts:
    for f in fs:
        if f is not None:
            x = x / f
    return x


def _reorder(x: Ts, dims: tuple[int, ...]) -> Ts:
    target_dims = tuple(i - len(dims) for i in range(len(dims)))
    if dims == target_dims:
        return x

    for target_dim, dim in zip(target_dims, dims):
        x = torch.transpose(x, dim, target_dim)
    return x


def _pad_in_dims(dims: tuple[int, ...], pad: tuple[int, ...], *tensors: Ts) -> list[Ts]:
    return [
        _reorder(functional.pad(_reorder(t, dims), pad, 'constant', 0), dims)
        for t in tensors
    ]


def _pad(t: Ts, dims: tuple[int, ...], pad: tuple[int, ...]) -> Ts:
    return _reorder(functional.pad(_reorder(t, dims), pad, 'constant', 0), dims)
