import torch
from torch import nn

from . import _surf
from ._surf import *
from ...base.typing import Any, Ts, Scalar, Vector, scalar, vector
from ... import mt

__all__ = [
    'build_surface',

    'Conic',
    'EvenAspherical',
    'Spherical',
    'Standard',
]
__all__ += _surf.__all__

EDGE_CUTTING: float = 1 - 1e-6


def _spherical(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    return c * r2 / (1 + torch.sqrt(1 - r2 * a))


def _spherical_der_wrt_r2(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    a = c.square() if k is None else c.square() * (1 + k)
    b = torch.sqrt(1 - r2 * a)
    return c / ((1 + b) * b)


class _SphericalBase(CircularSurface):
    def __init__(
        self, roc: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(radius, material, distance, newton_config)
        self.roc: nn.Parameter = nn.Parameter(scalar(roc))  #: Radius of curvature.

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        der = _spherical_der_wrt_r2(r2, 1 / self.roc) * 2
        return der * x, der * y

    @property
    def geo_radius(self) -> Ts:
        return self.roc * EDGE_CUTTING


class Spherical(_SphericalBase):
    r"""
    Spherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}

    where :math:`c` is radius of curvature.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    """

    # TODO: implement optimized intercept
    def __init__(
        self, roc: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, radius, material, distance, newton_config)


class Conic(_SphericalBase):
    r"""
    Conic surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}

    where :math:`c` is radius of curvature and :math:`k` is conic coefficient.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, radius, material, distance, newton_config)
        self.conic: nn.Parameter = nn.Parameter(scalar(conic))  #: Conic coefficient.

    def h_r2(self, r2: Ts) -> Ts:
        return _spherical(r2, 1 / self.roc, self.conic)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic) * 2
        return der * x, der * y

    @property
    def geo_radius(self) -> Ts:
        if self.conic.item() <= -1:
            return self.conic.new_tensor(float('inf'))
        else:
            return self.roc / torch.sqrt(1 + self.conic) * EDGE_CUTTING


class Standard(Conic):
    """Alias for :py:class:`~Conic`. This name complies with the convention of Zemax."""


class EvenAspherical(Conic):
    r"""
    Even aspherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}+\sum_{i=1}^N a_i r^{2i}

    where :math:`c` is radius of curvature, :math:`k` is conic coefficient
    and :math:`\{a_i\}_{i=1}^N` are even aspherical coefficients.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    :param coefficients: Even aspherical coefficients.
    :type coefficients: float, Sequence[float] or 1D Tensor
    """

    def __init__(
        self, roc: Scalar,
        conic: Scalar,
        coefficients: Vector,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(roc, conic, radius, material, distance, newton_config)
        self.coefficients: nn.Parameter = nn.Parameter(vector(coefficients))  #: Aspherical coefficients.

    def h_r2(self, r2: Ts) -> Ts:
        a = 0
        for c in reversed(self.coefficients):
            a = (a + c) * r2
        return super().h_r2(r2) + a

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        s_der = _spherical_der_wrt_r2(r2, 1 / self.roc, self.conic)
        a_der = 0
        for i in range(self.coefficients.numel(), 0, -1):
            a_der = a_der * r2 + self.coefficients[i - 1] * i
        der = (s_der + a_der) * 2
        return der * x, der * y

    @property
    def even_aspherical_items(self) -> int:
        """
        Number of even aspherical coefficients.

        :type: int
        """
        return self.coefficients.size(0)


def build_surface(surface_config: dict[str, Any]) -> Surface:
    raise NotImplementedError()
