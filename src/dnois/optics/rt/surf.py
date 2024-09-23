import torch
from torch import nn

from . import _surf
from ...base.typing import Any, Ts, Scalar, Vector, scalar, vector
from ... import mt
from ._surf import *

__all__ = [
    'build_surface',

    'EvenAspherical',
    'Surface',
]
__all__ += _surf.__all__

EDGE_CUTTING: float = 1 - 1e-5


def _spherical(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    return c * r2 / (1 + torch.sqrt(1 - r2 * a))


def _spherical_der_wrt_r2(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    b = torch.sqrt(1 - r2 * a)
    return c / ((1 + b) * b)


class _SphericalBase(Surface):
    def __init__(
        self, roc: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(radius, material, distance, newton_config)
        self.roc: nn.Parameter = nn.Parameter(scalar(roc))  #: Radius of curvature.

    def h(self, x: Ts, y: Ts = None) -> Ts:
        r2 = x if y is None else x.square() + y.square()
        return _spherical(r2, 1 / self.roc)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        der = _spherical_der_wrt_r2(r2, 1 / self.roc)
        der = der * 2
        return der * x, der * y

    @property
    def geo_radius(self) -> Ts:
        return self.roc * EDGE_CUTTING


class EvenAspherical(_SphericalBase):
    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        coefficients: Vector,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, radius, material, distance, newton_config)
        self.conic: nn.Parameter = nn.Parameter(scalar(conic))  #: Conic coefficient.
        self.coefficients: nn.Parameter = nn.Parameter(vector(coefficients))  #: Aspherical coefficients.

    def h(self, x: Ts, y: Ts = None) -> Ts:
        r2 = x if y is None else x.square() + y.square()
        a = 0
        for c in reversed(self.coefficients):
            a = (a + c) * r2
        return super().h(r2) + a

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        s_der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)
        a_der = 0
        for i in range(self.coefficients.numel(), 0, -1):
            a_der = a_der * r2 + self.coefficients[i - 1] * i
        der = (s_der + a_der) * 2
        return der * x, der * y


def build_surface(surface_config: dict[str, Any]) -> Surface:
    raise NotImplementedError()
