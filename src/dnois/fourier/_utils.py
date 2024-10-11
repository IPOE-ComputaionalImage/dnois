import torch

from dnois.base.typing import Ts, Spacing, Sequence


def _check_dim(name: str, shape: Sequence[int], dims: tuple[int, ...], **deltas: Spacing):
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
