import abc
import collections.abc

import torch
from torch import nn

from .ray import BatchedRay
from ... import mt, utils, torch as _t, base
from ...base import typing
from ...base.typing import Sequence, Ts, Any, Callable, Scalar, Self, Size2d

__all__ = [
    'NT_EPSILON',
    'NT_MAX_ITERATION',
    'NT_THRESHOLD',
    'NT_THRESHOLD_STRICT',
    'NT_UPDATE_BOUND',

    'surface_types',

    'Aperture',
    'BatchedRay',
    'CircularAperture',
    'CircularStop',
    'CircularSurface',
    'Context',
    'Planar',
    'Stop',
    'Surface',
    'SurfaceList',
]

NT_MAX_ITERATION: int = 10
NT_THRESHOLD: float = 20e-9
NT_THRESHOLD_STRICT: float = NT_THRESHOLD
NT_UPDATE_BOUND: float = 5.
NT_EPSILON: float = 1e-9

SAMPLE_LIMIT: float = 1 - 1e-4
EDGE_CUTTING: float = 1 - 1e-6


def _dist_transform(x: Ts, curve: Callable[[Ts], Ts]) -> Ts:
    magnitude = curve(x.abs())
    return magnitude.copysign(x)


def _rotation_mat(angles: Ts) -> Ts:
    c, s = angles.cos(), angles.sin()
    return torch.stack([
        torch.stack([c.prod() - s[2] * s[0], c[2] * c[1] * s[0] + s[2] * c[0], -c[2] * s[1]]),
        torch.stack([-s[2] * c[1] * c[0] - c[2] * s[0], -s[2] * c[1] * s[0] + c[2] * c[0], s[2] * s[1]]),
        torch.stack([s[1] * c[0], s[1] * s[0], c[1]]),
    ])


class Context(_t.EnhancedModule):
    """
    A class representing the context of a :py:class:`~Surface` in a list of surfaces. As component of the
    surface list, a surface does not hold the reference to the list but can access
    the information that depends on other surfaces in it via this class.
    Every surface contained in a surface list has a related context object.
    If it is not contained in any group, its context attribute is ``None``.

    This class also implements the conversion between global and surface-local coordinates.
    See :ref:`guide_optics_rt_slcs` for more details.

    :param Surface surface: The host surface that this context belongs to.
    :param SurfaceList surface_list: The surface list containing ``surface``.
    """
    x: Ts  #: x-coordinate of the origin of local coordinate.
    y: Ts  #: y-coordinate of the origin of local coordinate.
    z: Ts  #: z-coordinate of the origin of local coordinate.
    theta: Ts  #: Polar angle of z-axis of local coordinate.
    phi: Ts  #: Azimuthal angle of z-axis of local coordinate.
    chi: Ts  #: Spin angle of local coordinate.

    _transform_params = {'x', 'y', 'z', 'theta', 'phi', 'chi'}
    _writable_params = _t.EnhancedModule._writable_params | _transform_params

    def __init__(self, surface: 'Surface', surface_list: 'SurfaceList'):
        super().__init__()
        self.surface: 'Surface' = surface  #: The host surface that this context belongs to.
        self.surface_list: 'SurfaceList' = surface_list  #: The surface list containing the surface.

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        if name in self._transform_params:
            return self.new_tensor(0)  # default value for coordinate system parameters
        raise AttributeError(name)

    def __setattr__(self, key, value):
        if key in {'surface', 'surface_list'}:
            self.__dict__[key] = value  # avoid these two are registered as submodule
        else:
            return super().__setattr__(key, value)

    def g2l(self, x: Ts, direction: bool = False) -> Ts:
        r"""
        Converts global vectors ``x`` to local ones.
        If ``x`` represents positions, it is

        .. math::
            \mathbf{x}'=\mathbf{R}(\mathbf{x}-\mathbf{x}_0)

        where :math:`\mathbf{R}` is rotation matrix and :math:`\mathbf{x}_0` is :attr:`.origin`.
        If ``x`` represents directions, instead, it is

        .. math::
            \mathbf{x}'=\mathbf{R}\mathbf{x}

        :param Tensor x: Global vectors, a tensor of shape ``(..., 3)``.
        :param bool direction: Whether ``x`` represents directions. Default: ``False``.
        :return: Local vectors, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if self.shifted and not direction:
            x = x - self.origin
        if self.rotated:
            x = _rotation_mat(torch.stack([self.theta, self.phi, self.chi])) @ x.unsqueeze(-1)
        return x

    def l2g(self, x: Ts, direction: bool = False) -> Ts:
        r"""
        Converts local vectors ``x`` to global ones.
        If ``x`` represents positions, it is

        .. math::
            \mathbf{x}'=\mathbf{R}^{-1}\mathbf{x}+\mathbf{x}_0

        where :math:`\mathbf{R}` is rotation matrix and :math:`\mathbf{x}_0` is :attr:`.origin`.
        If ``x`` represents directions, instead, it is

        .. math::
            \mathbf{x}'=\mathbf{R}^{-1}\mathbf{x}

        :param Tensor x: Local vectors, a tensor of shape ``(..., 3)``.
        :param bool direction: Whether ``x`` represents directions. Default: ``False``.
        :return: Global vectors, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if self.rotated:
            x = _rotation_mat(-torch.stack([self.chi, self.phi, self.theta])).T @ x.unsqueeze(-1)
        if self.shifted and not direction:
            x = x + self.origin
        return x

    def g2l_ray(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.clone()
        ray.o = self.g2l(ray.o, False)
        ray.d = self.g2l(ray.d, True)
        return ray

    def l2g_ray(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.clone()
        ray.o = self.l2g(ray.o, False)
        ray.d = self.l2g(ray.d, True)
        return ray

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = {}
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            for name in self._transform_params:
                if name in params:
                    d[name] = self._attr2dictitem(name, keep_tensor)
        return d

    @property
    def index(self) -> int:
        """
        The index of the host surface in the surface list.

        :type: int
        """
        return self.surface_list.index(self.surface)

    @property
    def material_before(self) -> mt.Material:
        """
        :py:class:`~dnois.mt.Material` object before ths host surface.

        :type: :py:class:`~dnois.mt.Material`
        """
        idx = self.index
        if idx == 0:
            return self.surface_list.mt_head
        return self.surface_list[idx - 1].material

    @property
    def shifted(self) -> bool:
        """
        Whether the local coordinate system is shifted.

        :type: bool
        """
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            return any(n in params for n in 'xyz')
        return False

    @property
    def rotated(self) -> bool:
        """
        Whether the local coordinate system is rotated.

        :type: bool
        """
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            return any(n in params for n in ['theta', 'phi', 'chi'])
        return False

    @property
    def axis(self) -> Ts:
        r"""
        A unit vector of shape ``(3,)`` indicating rotated z axis in global coordinate system:

        .. math::
            \mathbf{A}=\left(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta\right)

        If assigning a tensor to it, it will be normalized to unit length automatically.

        :type: Tensor
        """
        s = self.theta.sin()
        return torch.stack([s * self.phi.cos(), s * self.phi.sin(), self.theta.cos()])

    @axis.setter
    def axis(self, value: Ts):
        _t.check_3d_vector(value, 'axis')
        if value.ndim != 1:
            raise base.ShapeError(f'axis must be a 1D vector, got shape {value.shape}')

        value: Ts = value / torch.linalg.vector_norm(value)
        theta = value[2].acos()
        phi = torch.atan2(value[1], value[0])
        self.theta = theta
        self.phi = phi

    @property
    def origin(self) -> Ts:
        r"""
        Coordinate of the origin of local coordinate system in the global one.
        A tensor of shape ``(3,)``. This property can be deleted to fix local origin
        to global origin.

        :type: Tensor
        """
        return torch.stack([self.x, self.y, self.z])

    @origin.setter
    def origin(self, value: Ts):
        _t.check_3d_vector(value, 'origin')
        if value.ndim != 1:
            raise base.ShapeError(f'origin must be a 1D vector, got shape {value.shape}')
        for i, n in enumerate('xyz'):
            setattr(self, n, value[i])

    @origin.deleter
    def origin(self):
        for n in 'xyz':
            delattr(self, n)

    @classmethod
    def from_dict(cls, d: dict) -> 'Context':
        raise TypeError(f'{cls.__name__} cannot be instantiated from a dict by calling {cls.from_dict.__qualname__}')

    def _check_available(self):
        if self.surface in self.surface_list:
            return
        raise RuntimeError(
            'The surface is not contained in the surface list referenced by its context object. '
            'This may be because the surface has been removed from the surface list.')


class CoaxialContext(Context):
    """
    A subclass of :class:`Context` for coaxial systems. In coaxial systems, default origins
    of local coordinate systems are arranged on z-axis and are determined by the distance
    from each surface to the next one. These points are called *baseline* s.
    Baseline of the first surface is fixed to 0. Shift and rotation of local coordinate systems
    can also be specified in order to, for example, simulate fabrication errors.

    Note that the ``origin`` of this class is defined relative to baseline. In other words,
    ``origin`` is ``(0, 0, 0)`` means the global coordinate of origin is ``(0, 0, baseline)``.

    See :class:`Context` for descriptions of more parameters.

    :param distance: Distance between baselines of the host surface and the next one.
    :type distance: float | Tensor
    """
    distance: nn.Parameter  #: Distance between baselines of the host surface and the next one.
    _writable_params = Context._writable_params | {'distance'}

    def __init__(
        self,
        surface: 'Surface',
        surface_list: 'SurfaceList',
        distance: typing.Scalar = None,
    ):
        super().__init__(surface, surface_list)
        self.register_parameter('distance', nn.Parameter(distance))

    def g2l(self, x: Ts, direction: bool = False) -> Ts:
        if not direction:
            x = x.clone()
            x[..., 2] -= self.baseline
        return super().g2l(x, direction)

    def l2g(self, x: Ts, direction: bool = False) -> Ts:
        x = super().l2g(x, direction)
        if not direction:
            x = x.clone()
            x[..., 2] += self.baseline
        return x

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['distance'] = self._attr2dictitem('distance', keep_tensor)
        return d

    @property
    def baseline(self) -> Ts:
        """
        The z-coordinate of the related surface's baseline. A 0D tensor.

        :type: Tensor
        """
        idx = self.index
        if idx == 0:
            return self.new_tensor(0.)
        z = self.surface_list[0].context.distance
        for s in self.surface_list[1:idx]:
            z = z + s.context.distance
        return z


class Aperture(_t.EnhancedModule, metaclass=abc.ABCMeta):
    """
    Base class for aperture shapes. Aperture refers to the region on a surface
    where rays can transmit. The region outside the aperture is assumed to be
    completely opaque. Note that the region inside is not necessarily completely
    transparent. The most common aperture type is :class:`CircularAperture`.

    Mathematically, an aperture is defined by a set of 2D points on the baseline
    plane of associated surface. See :class:`Surface` for more details.
    """

    @abc.abstractmethod
    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        """
        Returns a boolean tensor representing whether each point :math:`(x,y)` is
        inside the aperture. To jointly represent 2D coordinates, ``x`` and ``y``
        must be broadcastable.

        :param Tensor x: x coordinates of the points.
        :param Tensor y: y coordinates of the points.
        :return: See description above. The shape of returned tensor is the
            broadcast result of ``x`` and ``y``.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def sample_random(self, n: int) -> tuple[Ts, Ts]:
        """
        Returns ``n`` points randomly sampled on this aperture.

        :param int n: Number of points.
        :return: Two 1D tensors of length ``n``, representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def forward(self, ray: BatchedRay) -> BatchedRay:
        """
        Similar to :meth:`.evaluate`, but operates on rays.

        :param BatchedRay ray: Incident rays.
        :return: New rays among which those outside the aperture are marked as invalid.
        :rtype: BatchedRay
        """
        return ray.update_valid(self.pass_ray(ray))

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        """
        Similar to :meth:`.evaluate`, but operates on rays.

        :param BatchedRay ray: Incident rays.
        :return: A mask tensor indicating whether corresponding rays can pass the aperture.
        :rtype: torch.BoolTensor
        """
        return self.evaluate(ray.x, ray.y)

    def sample(self, mode: str, *args, **kwargs) -> tuple[Ts, Ts]:
        """
        Samples points on this aperture, i.e. baseline plane of associated surface.
        Specific distribution depends on ``mode``.

        :param str mode: Sampling mode. Calling object of this method should possess a
            ``sample_{mode}`` method.
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        meth_name = 'sample_' + mode
        meth: Callable = getattr(self, meth_name, ...)
        if meth is ...:
            raise ValueError(f'Unknown sampling mode for {self.__class__.__name__}: {mode}')
        return meth(*args, **kwargs)

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        return {'type': self.__class__.__name__}

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Aperture:
            d.pop('type')
            return cls(**d)  # default implementation of eponymous method

        ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == ty:
                return typing.cast(type[Aperture], sub).from_dict(d)  # Calling eponymous method of subclass
        aperture_types = [sub.__name__ for sub in subs]
        raise RuntimeError(f'Unknown aperture type: {ty}. Available: {aperture_types}')


class CircularAperture(Aperture):
    """
    Circular aperture with radius :attr:`radius`.

    :param diameter: Diameter of the aperture.
    :type diameter: float | Tensor
    """

    def __init__(self, diameter: Scalar):
        super().__init__()
        if diameter <= 0:
            raise ValueError('radius must be positive')

        self.register_buffer('radius', None)
        self.radius: Ts = typing.scalar(diameter) / 2  #: Radius of the aperture.

    def evaluate(self, x: Ts, y: Ts) -> torch.BoolTensor:
        return typing.cast(torch.BoolTensor, x.square() + y.square() <= self.radius.square())

    def pass_ray(self, ray: BatchedRay) -> torch.BoolTensor:
        return typing.cast(torch.BoolTensor, ray.r2 <= self.radius.square())

    def sample_random(self, n: int, sampling_curve: Callable[[Ts], Ts] = None) -> tuple[Ts, Ts]:
        r"""
        Returns ``n`` points randomly sampled on this aperture. An optional ``sampling_curve``
        (denoted by :math:`\Gamma`) can be specified to control the distribution of
        radial distance: :math:`r=\Gamma(t)` where :math:`t` is drawn uniformly from :math:`[0,1]`.

        :param int n: Number of points.
        :param sampling_curve: Sampling curve :math:`\Gamma(t)`. Default: :math:`\sqrt{t}`.
        :type sampling_curve: Callable[[Tensor], Tensor]
        :return: Two 1D tensors of length ``n``, representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        t = torch.rand(n, device=self.device, dtype=self.dtype) * (2 * torch.pi)
        r = torch.rand(n, device=self.device, dtype=self.dtype)
        if sampling_curve is not None:
            r = sampling_curve(r)
        else:
            r = r.sqrt()
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    def sample_rect(self, n: Size2d, mask_invalid: bool = True) -> tuple[Ts, Ts]:
        r"""
        Samples points on this aperture in a evenly spaced rectangular grid,
        where number of points in vertical and horizontal directions :math:`(H, W)`
        are given by ``n``. Note that the points outside the aperture are dropped
        so total number of returned points are is less than :math:`HW`.

        :param n: A pair of int representing :math:`(H, W)`.
        :type n: int | tuple[int, int]
        :param bool mask_invalid: Whether to discard points outside the aperture. Default: ``True``.
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        h, w = typing.size2d(n)
        y, x = utils.grid(
            (h, w), (2 * self.radius / h, 2 * self.radius / w), symmetric=True,
            device=self.device, dtype=self.dtype
        )
        y, x = torch.broadcast_tensors(y, x)
        x, y = x.flatten(), y.flatten()
        valid = self.evaluate(x, y)
        if mask_invalid:
            return x[valid], y[valid]
        else:
            return x, y

    def sample_unipolar(self, n: Size2d) -> tuple[Ts, Ts]:
        r"""
        Samples points on this aperture in a unipolar manner. Specifically, the aperture
        is divided into :math:`N_r` rings with equal widths and points are sampled on the
        outer edge of each ring. The first ring contains :math:`N_\theta` points, the second
        contains :math:`2N_\theta` points ... and so on, plus a point at center.
        Thus, there are :math:`N_\theta N_r(N_r+1)/2+1` points in total.

        :param n: A pair of int representing :math:`(N_r,N_\theta)`.
        :type n: int | tuple[int, int]
        :return: Two 1D tensors of representing x and y coordinates of the points.
        :rtype: tuple[Tensor, Tensor]
        """
        n = typing.size2d(n)
        zero = torch.tensor([0.], dtype=self.dtype, device=self.device)
        r = torch.linspace(0, self.radius * SAMPLE_LIMIT, n[0] + 1, device=self.device, dtype=self.dtype)
        r = [r[i].expand(i * n[1]) for i in range(1, n[0] + 1)]  # n[1]*n[0]*(n[0]+1)/2
        r = torch.cat([zero] + r)  # n[1]*n[0]*(n[0]+1)/2+1
        t = [
            torch.arange(i * n[1], device=self.device, dtype=self.dtype) / (n[1] * i) * (2 * torch.pi)
            for i in range(1, n[0] + 1)
        ]
        t = torch.cat([zero] + t)
        return r * t.cos(), r * t.sin()

    def sample_diameter(self, n: int, theta: float | Ts) -> tuple[Ts, Ts]:
        """
        Samples points on diameter line segments of this aperture.
        Polar angle of the line is given by ``theta``.

        :param int n: Number of points.
        :param theta: Polar angle of the line. A single float or a tensor with any shape.
        :type theta: float | Tensor
        :return: Two tensors representing x and y coordinates of the points.
            If ``theta`` is a float, with shape ``(n,)``; if a tensor with shape ``(...)``,
            with shape ``(..., n)``.
        """
        if not torch.is_tensor(theta):
            theta = torch.tensor(theta, dtype=self.dtype, device=self.device)
        r = torch.linspace(-1, 1, n, device=self.device, dtype=self.dtype) * self.radius
        theta = theta.unsqueeze(-1)
        return r * theta.cos(), r * theta.sin()

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['radius'] = self._attr2dictitem('radius', keep_tensor)
        return d


# TODO: ray validity check
class Surface(_t.EnhancedModule, metaclass=abc.ABCMeta):
    r"""
    Base class for optical surfaces in a group of lens.

    The geometric shape of a surface is described by an equation in
    :ref:`surface-local coordinate system <guide_optics_rt_slcs>`
    :math:`z=h(x,y)`, which has different forms for each surface type.
    The function :math:`h`, called *surface function*,
    is a 2D function of lateral coordinates :math:`(x,y)` which satisfies :math:`h(0,0)=0`.
    Note that the surface function also depends on the parameters of the surface implicitly.

    To ensure that a ray propagating along z-axis must have an intersection with
    the surface, an extended surface function (see :py:meth:`~h_extended`)
    is computed to find the intersection. Normally, the definition domain of
    surface function covers the aperture so an extended surface does not
    affect actual surface. If it cannot cover the aperture, however, the
    actual surface will be extended, which is usually undesired.

    This is a subclass of :py:class:`torch.nn.Module`.

    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param Aperture aperture: :class:`Aperture` of this surface.
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """

    def __init__(
        self,
        material: mt.Material | str,
        aperture: Aperture,
        reflective: bool = False,
        newton_config: dict[str, Any] = None,
        *,
        d: Scalar = None,
    ):
        super().__init__()
        if newton_config is None:
            newton_config = {}
        #: Material following the surface.
        self.material: mt.Material = material if isinstance(material, mt.Material) else mt.get(material)
        #: :class:`Aperture` of this surface.
        self.aperture: Aperture = aperture
        #: The context object of the surface in a surface list.
        #: This is created by the surface list object containing the surface.
        self.context: Context | None = None
        #: Whether this surface reflects (rather than refracts) rays.
        self.reflective: bool = reflective

        if d is not None:
            self._distance = typing.scalar(d, dtype=self.dtype, device=self.device)

        self._nt_max_iteration = newton_config.get('max_iteration', NT_MAX_ITERATION)
        self._nt_threshold = newton_config.get('threshold', NT_THRESHOLD)
        self._nt_threshold_strict = newton_config.get('threshold_strict', NT_THRESHOLD_STRICT)
        self._nt_update_bound = newton_config.get('update_bound', NT_UPDATE_BOUND)
        self._nt_epsilon = newton_config.get('epsilon', NT_EPSILON)

    @abc.abstractmethod
    def h(self, x: Ts, y: Ts) -> Ts:
        r"""
        Computes surface function :math:`h(x,y)`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of the surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes the partial derivatives of surface function
        :math:`\pfrac{h(x,y)}{x}` and :math:`\pfrac{h(x,y)}{y}`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    @abc.abstractmethod
    def h_extended(self, x: Ts, y: Ts) -> Ts:
        r"""
        Computes extended surface function:

        .. math::

            \tilde{h}(x,y)=\left\{\begin{array}{ll}
                h(x,y) & \text{if}(x,y)\in\text{dom} h,\\
                \text{extended value} & \text{else}
            \end{array}\right.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of extended surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad_extended(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes partial derivatives of extended surface function:
        :math:`\pfrac{\tilde{h}(x,y)}{x}` and :math:`\pfrac{\tilde{h}(x,y)}{y}`.
        See :py:meth:`~h_extended`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        """
        Returns the refracted rays of a group of incident rays ``ray``.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = self.intercept(ray)
        if self.reflective:
            return self.reflect(ray)
        else:
            return self.refract(ray, forward)

    def intercept(self, ray: BatchedRay) -> BatchedRay:
        """
        Returns a new :py:class:`~BatchedRay` whose directions are identical to those
        of ``ray`` and origins are the intersections of ``ray`` and this surface.
        The intersections are solved by `Newton's method
        <https://en.wikipedia.org/wiki/Newton's_method>`_ .
        The rays for which no intersection with sufficient precision, within the aperture
        and resulted from a positive marching distance will be marked as invalid.

        :param BatchedRay ray: Incident rays.
        :return: Intercepted rays.
        :rtype: BatchedRay
        """
        ray = ray.norm_d()
        t = self._solve_t(self.context.g2l_ray(ray))
        ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))

        ray_in_local = self.context.g2l_ray(ray)
        ray.update_valid_(
            self.aperture.pass_ray(ray_in_local) &
            (self._f(ray_in_local).abs() < self._nt_threshold_strict)
        )
        return ray

    def normal(self, x: Ts, y: Ts) -> Ts:
        """
        Returns unit normal vector of the surface pointing to positive-z direction,
        i.e., from before the surface to behind it.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :return: A tensor whose shape depends on ``x`` and ``y``, with an additional
            dimension of size 3 following.
        :rtype: Tensor
        """
        phpx, phpy = self.h_grad_extended(x, y)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are refracted by this surface.
        See :meth:`dnois.refract` for more details.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        if self.context.material_before == self.material:
            return ray

        normal = self._optical_normal(ray.x, ray.y)
        normal = self.context.l2g(normal, True)
        if forward:
            mu = self.context.material_before.n(ray.wl, 'm') / self.material.n(ray.wl, 'm')
        else:
            mu = self.material.n(ray.wl, 'm') / self.context.material_before.n(ray.wl, 'm')
            normal = -normal
        refractive = base.refract(ray.d_norm, normal, mu)
        ray.d = refractive.nan_to_num(nan=0)
        ray.update_valid_(~refractive[..., 0].isnan()).norm_d_()
        return ray

    def reflect(self, ray: BatchedRay) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are reflected by this surface.
        See :meth:`dnois.reflect` for more details.

        :param BatchedRay ray: Incident rays.
        :return: Reflected rays with origin on this surface. A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        normal = self._optical_normal(ray.x, ray.y)
        normal = self.context.l2g(normal, True)
        ray.d = base.reflect(ray.d_norm, normal)
        return ray

    def sample(self, mode: str, *args, **kwargs) -> Ts:
        """
        Samples points on this surface. They are first sampled by :meth:`Aperture.sample`.

        :param str mode: Sampling mode. See :meth:`Aperture.sample`.
        :return: A tensor with shape ``(n, 3)`` where ``n`` is number of samples.
            ``3`` means 3D spatial coordinates.
        :rtype: Tensor
        """
        x, y = self.aperture.sample(mode, *args, **kwargs)
        z = self.h_extended(x, y)
        points = torch.stack([x, y, z], dim=-1)
        points = self.context.l2g(points)
        return points

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'material': self.material.name,
            'aperture': self.aperture.to_dict(keep_tensor),
        }

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Surface:
            d.pop('type')
            d['aperture'] = Aperture.from_dict(d['aperture'])
            return cls(**d)  # default implementation of eponymous method

        ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == ty:
                return typing.cast(type[Surface], sub).from_dict(d)  # calling eponymous method of subclass
        raise RuntimeError(f'Unknown surface type: {ty}. Available: {surface_types(True)}')

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_extended(ray.x, ray.y) - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent

    def _solve_t(self, ray: BatchedRay) -> Ts:
        # the origin and direction of ray are defined in local coordinate system
        ray = ray.norm_d()
        t = - ray.z / ray.d_z
        t0 = t
        cnt = 0  # equal to numbers of derivative computation
        new_ray = ray.clone(False)
        with torch.no_grad():
            while True:
                new_ray.o = ray.o + new_ray.d_norm * t.unsqueeze(-1)  # do not compute opl for root finder
                f_value = self._f(new_ray)
                if torch.all(f_value.abs() < self._nt_threshold) or cnt >= self._nt_max_iteration:
                    break

                t = t - self._newton_descent(new_ray, f_value)
                cnt += 1

            t = t - t0  # trace back

        t = t + t0  # this is needed to compute gradient correctly
        new_ray.o = ray.o + new_ray.d_norm * t.unsqueeze(-1)

        # the second argument cannot be replaced by f_value because of computational graph
        return t - self._newton_descent(new_ray, self._f(new_ray))

    # for surfaces like Fresnel
    def _optical_normal(self, x: Ts, y: Ts) -> Ts:
        return self.normal(x, y)


class Planar(Surface):
    def __init__(
        self,
        material: mt.Material | str,
        aperture: Aperture,
        reflective: bool = False,
        *,
        d: Scalar = None
    ):
        super().__init__(material, aperture, reflective, None, d=d)

    def h(self, x: Ts, y: Ts) -> Ts:
        return self.new_zeros(torch.broadcast_shapes(x.shape, y.shape))

    def h_extended(self, x: Ts, y: Ts) -> Ts:
        return self.new_zeros(torch.broadcast_shapes(x.shape, y.shape))

    def h_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def h_grad_extended(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def intercept(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.norm_d()
        t = self._solve_t(self.context.g2l_ray(ray))
        ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))

        ray_in_local = self.context.g2l_ray(ray)
        ray.update_valid_(self.aperture.pass_ray(ray_in_local))
        return ray

    def normal(self, x: Ts, y: Ts) -> Ts:
        return torch.stack([torch.zeros_like(x), torch.zeros_like(y), torch.ones_like(x)], -1)

    def _solve_t(self, ray: BatchedRay) -> Ts:
        ray = ray.norm_d()
        return - ray.z / ray.d_z


class Stop(Planar):
    def __init__(self, aperture: Aperture, move_ray: bool = True, *, d: Scalar = None):
        super().__init__('vacuum', aperture, False, d=d)  # material is ignored
        self._move_ray = move_ray

    def intercept(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.norm_d()
        ray_in_local = self.context.g2l_ray(ray)
        t = self._solve_t(ray_in_local)
        if self._move_ray:
            ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))
            ray_in_local = self.context.g2l_ray(ray)
            ray.update_valid_(self.aperture.pass_ray(ray_in_local))
            return ray
        else:
            new_o = ray_in_local.o + t.unsqueeze(-1) * ray_in_local.d_norm
            valid_ap = self.aperture.evaluate(new_o[..., 0], new_o[..., 1])
            return ray.update_valid(valid_ap)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        return ray.clone()

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        del d['material']
        return d

    @property
    def material(self) -> mt.Material:  # override
        return self.context.material_before

    @material.setter
    def material(self, value):
        pass


class CircularSurface(Surface, metaclass=abc.ABCMeta):
    r"""
    Derived class of :py:class:`~Surface` for optical surfaces
    with circular symmetry, i.e. its property
    depends only on the radial distance :math:`r=\sqrt{x^2+y^2}`, in a group of lens.
    Therefore, their surface function can be written as
    :math:`h(x,y)=\hat{h}(x^2+y^2)=\hat{h}(r^2)`.
    Note that :math:`\hat{h}`
    takes as input squared radial distance for computational efficiency purpose.

     Despite the circular symmetry of the surface, its aperture is not necessarily
     circularly symmetric. In other words, ``aperture`` need not be an instance of
     :class:`CircularAperture`.

    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param aperture: Aperture of this surface. If a float, the aperture will be a
        :class:`CircularAperture` whose diameter is the given value. Default: infinity.
    :type aperture: Aperture or float
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """

    def __init__(
        self,
        material: mt.Material | str,
        aperture: Aperture | float = None,
        reflective: bool = False,
        newton_config: dict[str, Any] = None,
        *,
        d: Scalar = None
    ):
        if aperture is None:
            aperture = float('inf')
        if isinstance(aperture, float):
            aperture = CircularAperture(aperture)
        super().__init__(material, aperture, reflective, newton_config, d=d)

    @abc.abstractmethod
    def h_r2(self, r2: Ts) -> Ts:
        r"""
        Computes surface function :math:`\hat{h}(r^2)`.

        :param Tensor r2: Squared radial distance.
        :return: Corresponding value of the surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_derivative_r2(self, r2: Ts) -> Ts:
        r"""
        Computes derivative :math:`\frac{\d\hat{h}(r^2)}{\d r^2}`.

        :param Tensor r2: Squared radial distance.
        :return: Corresponding value of the derivative.
        :rtype: Tensor
        """
        pass

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Computes the partial derivatives of surface function
        :math:`\pfrac{h(x,y)}{x}` and :math:`\pfrac{h(x,y)}{y}`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared radial distance. It can be passed in to avoid
            repeated computation if already computed outside this method.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        if r2 is None:
            r2 = x.square() + y.square()
        derivative_double = self.h_derivative_r2(r2) * 2
        return derivative_double * x, derivative_double * y

    def h(self, x: Ts, y: Ts) -> Ts:
        return self.h_r2(x.square() + y.square())

    def h_extended(self, x: Ts, y: Ts) -> Ts:
        return self.h_r2_extended(x.square() + y.square())

    def h_grad_extended(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Computes partial derivatives of extended surface function:
        :math:`\pfrac{\tilde{h}(x,y)}{x}` and :math:`\pfrac{\tilde{h}(x,y)}{y}`.
        See :py:meth:`~h_extended`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared radial distance. It can be passed in to avoid
            repeated computation if already computed outside this method.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        if r2 is None:
            r2 = x.square() + y.square()
        lim2 = self.geo_radius.square()
        phpx, phpy = self.h_grad(x, y, r2)
        if lim2.isinf().all():
            return phpx, phpy
        mask = r2 <= lim2
        return torch.where(mask, phpx, 0), torch.where(mask, phpy, 0)

    def h_r2_extended(self, r2: Ts) -> Ts:
        r"""
        Computes extended version of :math:`\hat{h}(r^2)`.
        See :py:meth:`~h_r2` and :py:meth:`~h_extended`.
        """
        lim2 = self.geo_radius.square()
        if lim2.isinf().all():
            return self.h_r2(r2)
        return torch.where(r2 <= lim2, self.h_r2(r2), self.h_r2(lim2 * EDGE_CUTTING))

    @property
    def geo_radius(self) -> Ts:
        """
        Geometric radius of the surface, i.e. maximum radial distance that makes
        the surface function mathematically meaningful. A 0D tensor.

        :type: Tensor
        """
        return self.new_tensor(float('inf'))

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_r2_extended(ray.r2) - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y, ray.r2)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    # def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
    #     derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
    #     descent = f_value / (derivative_value + self._nt_epsilon)
    #     descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
    #     return descent


class CircularStop(Stop, CircularSurface):
    def __init__(self, aperture: Scalar, *, d: Scalar = None):
        if isinstance(aperture, float):
            aperture = CircularAperture(aperture)
        super().__init__(aperture, d=d)

    def h_derivative_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    def h_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)


class SurfaceList(nn.ModuleList, collections.abc.MutableSequence, base.AsJsonMixIn):
    """
    A sequential container of surfaces. This class is derived from
    :py:class:`torch.nn.ModuleList` and implements
    :py:class:`collections.abc.MutableSequence` interface.
    So its instance can be regarded as both a PyTorch module
    and a list of :py:class:`Surface`.

    :param surfaces: A sequence of :py:class:`Surface` objects. Default: ``[]``.
    :type surfaces: Sequence[Surface]
    :param foremost_material: The material before the first surface.
    :type foremost_material: :py:class:`~dnois.mt.Material`
    """
    _force_surface: bool = True
    __call__: Callable[..., BatchedRay]  # for return type hint in IDE

    def __init__(
        self,
        surfaces: Sequence[Surface] = None,
        foremost_material: mt.Material | str = 'vacuum',
        coaxial: bool = True,
    ):
        super().__init__()
        if surfaces is None:
            surfaces = []
        if not isinstance(foremost_material, mt.Material):
            foremost_material = mt.get(foremost_material)

        # This is needed to facilitate MutableSequence operations
        # because torch.nn.ModuleList saves submodules like a dict rather than list
        self._slist: list[Surface] = []
        self._stop_idx = None
        #: Material before the first surface.
        self.mt_head: mt.Material = foremost_material
        #: Whether the surfaces are coaxial.
        self.coaxial: bool = coaxial

        self.extend(surfaces)

    def __contains__(self, item) -> bool:
        """:meta private:"""
        return self._slist.__contains__(item)

    def __delitem__(self, key):
        """:meta private:"""
        super().__delitem__(key)
        self._slist.__getitem__(key).context = None
        self._slist.__delitem__(key)

    def __getitem__(self, item) -> Surface | list[Surface]:
        """:meta private:"""
        return self._slist.__getitem__(item)

    def __iadd__(self, other: Sequence[Surface]) -> Self:
        """:meta private:"""
        self.extend(other)
        return self

    def __iter__(self):
        """:meta private:"""
        return self._slist.__iter__()

    def __len__(self):
        """:meta private:"""
        return self._slist.__len__()

    def __reversed__(self) -> 'SurfaceList':
        """:meta private:"""
        return SurfaceList(list(self._slist.__reversed__()), self.mt_head)

    def __setitem__(self, key: int, value: Surface):
        """:meta private:"""
        self._welcome(value)
        super().__setitem__(key, value)
        self._slist.__setitem__(key, value)

    def __add__(self, other: Sequence[Surface]) -> 'SurfaceList':
        """:meta private:"""
        return SurfaceList(self._slist + list(other), self.mt_head)

    def __repr__(self) -> str:
        """:meta private:"""
        _repr = super().__repr__()[:-1]  # remove that last parentheses
        _repr += f'  env_material={repr(self.mt_head)}\n)'
        return _repr

    def __dir__(self):
        """:meta private:"""
        return super().__dir__() + ['env_material']

    def append(self, surface: Surface):
        """:meta private:"""
        self._welcome(surface)
        super().append(surface)

    def clear(self):
        """:meta private:"""
        for s in self._slist:
            s.context = None
        self._slist.clear()
        self._super_clear()

    def count(self, value: Surface) -> int:
        """:meta private:"""
        return self._slist.count(value)

    def extend(self, surfaces: Sequence[Surface]):
        """:meta private:"""
        self._welcome(*surfaces)
        super().extend(surfaces)

    def index(self, value: Surface, start: int = 0, stop: int = ...) -> int:
        """:meta private:"""
        if stop is ...:
            return self._slist.index(value, start)
        else:
            return self._slist.index(value, start, stop)

    def insert(self, index: int, surface: Surface):
        """:meta private:"""
        self._welcome(surface)
        super().insert(index, surface)
        self._slist.insert(index, surface)

    def pop(self, index: int = -1) -> Surface:
        """:meta private:"""
        s = super().pop(index)
        s.context = None
        return s

    def remove(self, value: Surface):
        """:meta private:"""
        idx = self.index(value)
        self.pop(idx)

    def reverse(self):
        """:meta private:"""
        ss = self._slist.copy()
        self._slist.clear()
        self._super_clear()
        self.extend(ss)

    def add_module(self, name: str, module: nn.Module):
        """:meta private:"""
        if name.isdigit() and isinstance(module, Surface):
            self._slist.insert(int(name), module)
        super().add_module(name, module)

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        """
        Traces rays incident on the first surface and returns rays
        passing the last surface, or reversely if ``forward`` is ``False``.

        :param BatchedRay ray: Input rays.
        :param bool forward: Whether rays are forward or not.
        :return: Output rays.
        :rtype: BatchedRay
        """
        for s in (self._slist if forward else reversed(self._slist)):
            try:
                ray = s(ray, forward)
            except Exception as e:
                idx = self.index(s)
                e.add_note(f'This exception is raised during the forward pass of surface {idx}')
                raise e
        return ray

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        return {
            'surfaces': [s.to_dict(keep_tensor) for s in self._slist],
            'contexts': [s.context.to_dict(keep_tensor) for s in self._slist],
            'foremost_material': self.mt_head.name,
            'coaxial': self.coaxial,
        }

    @property
    def first(self) -> Surface:
        """
        Returns the first surface.

        :type: :class:`Surface`.
        """
        return self[0]

    @property
    def last(self) -> Surface:
        """
        Returns the last surface.

        :type: :class:`Surface`.
        """
        return self[-1]

    @property
    def is_empty(self) -> bool:
        """
        Whether this list is empty.

        :type: bool
        """
        return len(self._slist) == 0

    @property
    def length(self) -> Ts:
        """
        Returns the distance between baselines of the first and that of the last surfaces
        as a 0D tensor.

        :type: Tensor
        """
        return typing.cast(Ts, sum(s.context.distance for s in self._slist[:-1]))

    @property
    def mt_tail(self) -> mt.Material:
        """
        Returns the material after the last surface.

        :type: :class:`~Material`
        """
        return self._slist[-1].material

    @property
    def total_length(self) -> Ts:
        """
        Returns the sum of :py:attr:`Surface.distance` of all the surfaces
        as a 0D tensor.

        :type: Tensor
        """
        return typing.cast(Ts, sum(s.context.distance for s in self._slist))

    @property
    def stop_idx(self) -> int | None:
        """
        Index of the aperture stop. Returns ``None`` if no stop is found.

        :type: int or ``None``
        """
        return self._stop_idx

    @property
    def stop(self) -> CircularStop | None:
        """
        The aperture stop object. Returns ``None`` if no stop is found.
        Note that it need not return an instance of :py:class:`CircularStop`.

        :type: :py:class:`CircularStop` or ``None``
        """
        idx = self._stop_idx
        return None if idx is None else self._slist[idx]

    @classmethod
    def from_dict(cls, d: dict):
        d['surfaces'] = [Surface.from_dict(s) for s in d['surfaces']]
        sl = cls(**d)
        for i, ctx in enumerate(d['contexts']):
            for k, v in ctx.items():
                setattr(sl[i].context, k, v)
        return sl

    def _welcome(self, *new: Surface):
        for surface in new:
            if self._force_surface and not isinstance(surface, Surface):
                raise TypeError(f'An instance of {Surface.__name__} expected, got {type(surface).__name__}')
            if self.coaxial:
                surface.context = CoaxialContext(surface, self, getattr(surface, '_distance', None))
            else:
                surface.context = Context(surface, self)

        for s1 in self._slist:
            for s2 in new:
                if id(s1) == id(s2):
                    raise ValueError('Trying to add a surface into a surface list containing it')

    def _super_clear(self):
        for idx in range(len(self) - 1, -1, -1):
            super().__delitem__(idx)


def surface_types(name_only: bool = False) -> list[type[Surface]] | list[str]:
    """
    Returns a list of accessible subclasses of :class:`Surface` in lexicographic order.
    This can be used to recognize surface types supported by dnois.

    :param bool name_only: If ``True``, returns class names, otherwise returns class objects.
    :return: A list of subclasses of :class:`Surface`.
    :rtype: list[type[Surface]] or list[str]
    """
    sub_list = utils.subclasses(Surface)
    if name_only:
        return [sub.__name__ for sub in sub_list]
    else:
        return typing.cast(list, sub_list)
