import torch

from ...base import typing
from ...base.typing import Ts, Size2d

__all__=[
    'max_center',
    'min_peripheral',
    'min_peripheral_var',
]


def max_center(psf: Ts, target: float | Ts = 1, center: Size2d = None) -> Ts:
    if center is None:
        center = (psf.size(-2) // 2, psf.size(-1) // 2)
    center = typing.size2d(center)

    v = torch.square(psf[..., *center] - target)  # ...
    return v.mean()


def min_peripheral(psf: Ts, center: Size2d = None) -> Ts:
    if center is None:
        center = (psf.size(-2) // 2, psf.size(-1) // 2)
    center = typing.size2d(center)
    center = center[0] * psf.size(-2) + center[1]

    psf = psf.flatten(-2, -1)  # ... x (HW)
    psf = psf[..., :center].square().sum(-1) + psf[..., center + 1:].square().sum(-1)  # ...
    return psf.mean()


def min_peripheral_var(psf: Ts, center: Size2d = None) -> Ts:
    if center is None:
        center = (psf.size(-2) // 2, psf.size(-1) // 2)
    center = typing.size2d(center)
    center = center[0] * psf.size(-2) + center[1]

    psf = psf.flatten(-2, -1)  # ... x (HW)
    seg1, seg2 = psf[..., :center], psf[..., center + 1:]
    n = seg1.size(-1) + seg2.size(-1)
    mean = torch.unsqueeze((seg1.sum(-1) + seg2.sum(-1)) / n, -1)  # ... x  1
    var = ((seg1 - mean).square().sum(-1) + (seg2 - mean).square().sum(-1)) / n  # ...
    return var.mean()
