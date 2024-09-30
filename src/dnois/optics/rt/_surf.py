import abc
import collections.abc

import torch
from torch import nn

from .ray import BatchedRay
from ... import mt, base
from ...base.typing import (
    Sequence, Ts, Any, Callable, Scalar, Self, Size2d, SurfSample,
    scalar, overload, size2d,
)

__all__ = [
    'NT_EPSILON',
    'NT_MAX_ITERATION',
    'NT_THRESHOLD',
    'NT_THRESHOLD_STRICT',
    'NT_UPDATE_BOUND',

    'CircularSurface',
    'Context',
    'Surface',
    'SurfaceList',
]

NT_MAX_ITERATION: int = 10
NT_THRESHOLD: float = 20e-9
NT_THRESHOLD_STRICT: float = 20e-9
NT_UPDATE_BOUND: float = 5.
NT_EPSILON: float = 1e-9

SAMPLE_LIMIT: float = 1 - 1e-4
EDGE_CUTTING: float = 1 - 1e-5


class Context:
    """
    A class representing the context of a :py:class:`~Surface` in a lens group. As component of the
    lens group, a surface does not hold the reference to the group but can access
    the information that depends on other surfaces in it via this class.
    Every surface contained in a lens group has a related context object.
    If it is not contained in any group, its context attribute is ``None``.

    :param CircularSurface surface: The host surface that this context belongs to.
    :param SurfaceList lens_group: The lens group containing the surface.
    """

    def __init__(self, surface: 'CircularSurface', lens_group: 'SurfaceList'):
        self.surface: 'CircularSurface' = surface  #: The host surface that this context belongs to.
        self.lens_group: 'SurfaceList' = lens_group  #: The lens group containing the surface.

    @property
    def index(self) -> int:
        """
        The index of the host surface in the lens group.

        :type: int
        """
        return self.lens_group.index(self.surface)

    @property
    def material_before(self) -> mt.Material:
        """
        :py:class:`~dnois.mt.Material` object before ths host surface.

        :type: :py:class:`~dnois.mt.Material`
        """
        idx = self.index
        if idx == 0:
            return self.lens_group.env_material
        return self.lens_group[idx - 1].material

    @property
    def z(self) -> Ts:
        """
        The z-coordinate of the related surface's baseline. A 0D tensor.

        :type: Tensor
        """
        idx = self.index
        if idx == 0:
            return self.surface.distance.new_tensor(0.)
        z = self.lens_group[0].distance
        for s in self.lens_group[1:idx]:
            z = z + s.distance
        return z

    def _check_available(self):
        if self.surface in self.lens_group:
            return
        raise RuntimeError(
            'The surface is not contained in the lens group referenced by its context object. '
            'This may be because the surface has been removed from the lens group.')


class Surface(nn.Module, base.TensorContainerMixIn, metaclass=abc.ABCMeta):
    r"""
    Base class for optical surfaces in a group of lens.
    The position of a surface in a lens group is specified by a single z-coordinate,
    which is called its *baseline*. Baseline of the first surface is 0.

    The geometric shape of a surface is described by an equation
    :math:`z=h(x,y)+z_\text{baseline}`, which has different forms
    for each surface type. The function :math:`h`, called *surface function*,
    is a 2D function of lateral coordinates :math:`(x,y)` which satisfies :math:`h(0,0)=0`.
    Note that the surface function also depends on the parameters of the surface implicitly.

    To ensure that a ray propagating along z-axis must have an intersection with
    the surface, an extended surface function (see :py:meth:`~h_extended`)
    is computed to find the intersection. Normally, the definition domain of
    surface function covers the aperture so an extended surface does not
    affect actual surface. If it cannot cover the aperture, however, the
    actual surface will be extended, which is usually undesired.
    At present a warning will be issued if that case is detected.

    This is subclass of :py:class:`torch.nn.Module`.

    .. note::

        All the quantities with length dimension is represented in meters.
        They include value of coordinates, optical path length and wavelength, etc.

    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param distance: Distance between the surface and the next one.
    :type distance: float or Tensor
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """
    _delegate_name = 'distance'

    def __init__(
        self,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__()
        distance = scalar(distance)
        if distance.item() < 0:
            raise ValueError('distance must not be negative')
        if newton_config is None:
            newton_config = {}
        #: Material following the surface.
        self.material: mt.Material = material if isinstance(material, mt.Material) else mt.get(material)
        #: Distance between the surface and the next one.
        #: This is an optimizable parameter by default.
        self.distance: nn.Parameter = nn.Parameter(distance)
        #: The context object of the surface in a lens group.
        #: This is created by the lens group object containing the surface.
        self.context: Context | None = None

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

    @overload
    def _valid(self, x: Ts, y: Ts) -> Ts:
        pass

    @overload
    def _valid(self, ray: BatchedRay) -> Ts:
        pass

    @abc.abstractmethod
    def _valid(self, *args, **kwargs) -> Ts:
        """
        Checks whether given rays whose origins are on the surface are valid.

        :param BatchedRay ray: Rays to be checked.
        :return: Flags of validity.
        :rtype: torch.BoolTensor
        """
        pass

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        """
        Returns the refracted rays of a group of incident rays ``ray``.
        The directions of rays are determined by ``forward`` and the rays
        with incorrect direction will be marked as invalid.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        magnitude = self._nt_threshold_strict / torch.finfo(ray.o.dtype).eps
        if torch.abs(ray.z - self.context.z).gt(magnitude).any():
            new_z = self.context.z - magnitude if forward else self.context.z + magnitude
            ray.march_to_(new_z, self.context.material_before.n(ray.wl, 'm'))

        if forward:
            valid = torch.logical_and(self._f(ray) >= 0, ray.d_z > 0)
        else:
            valid = torch.logical_and(self._f(ray) <= 0, ray.d_z < 0)
        ray.update_valid_(valid, 'copy')

        return self.refract(self.intercept(ray), forward)

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
        # TODO: optimize
        t = (self.context.z - ray.z) / ray.d_z
        cnt = 0  # equal to numbers of derivative computation
        new_ray = ray.clone(False)
        new_ray.norm_d_()
        with torch.no_grad():
            while True:
                new_ray.o = ray.o + new_ray.d_norm * t.unsqueeze(-1)  # do not compute opl for root finder
                f_value = self._f(new_ray)
                if torch.all(f_value.abs() < self._nt_threshold) or cnt >= self._nt_max_iteration:
                    break

                t = t - self._newton_descent(new_ray, f_value)
                cnt += 1

        # the second argument cannot be replaced by f_value because of computational graph
        t = t - self._newton_descent(new_ray, self._f(new_ray))
        ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))
        ray.update_valid_(
            self._valid(ray) &
            (self._f(ray).abs() < self._nt_threshold_strict) &
            (t > 0), 'copy'
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
        phpx, phpy = self.h_grad(x, y)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are refracted by this surface.
        Refracted directions are computed by following equation:

        .. math::

            \mathbf{d}_2=\mu\mathbf{d}_1+\left(
                \sqrt{1-\mu^2\left(1-(\mathbf{n}\cdot\mathbf{d}_1)^2\right)}-
                \mu\mathbf{n}\cdot\mathbf{d}_1
            \right)\mathbf{n}

        where :math:`\mathbf{d}_1` and :math:`\mathbf{d}_2` are unit vectors of
        incident and refracted directions, :math:`\mu=n_1/n_2` is the ratio of
        refractive indices and :math:`\mathbf{n}` is unit normal vector of the surface,
        which is exactly :py:meth:`~normal`. Note that this equation applies only
        to forward ray tracing.
        The rays making the expression under the square root negative will be
        marked as invalid.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        n = self.normal(ray.x, ray.y)
        i = ray.d_norm
        if forward:
            miu = self.context.material_before.n(ray.wl, 'm') / self.material.n(ray.wl, 'm')
        else:
            miu = self.material.n(ray.wl, 'm') / self.context.material_before.n(ray.wl, 'm')
            n = -n
        miu = miu.unsqueeze(-1)
        ni = torch.sum(n * i, dim=-1).unsqueeze(-1)
        nt2 = 1 - miu.square() * (1 - ni.square())
        t = torch.sqrt(nt2.relu_()) * n + miu * (i - ni * n)
        ray.d = t
        ray.update_valid_(nt2.squeeze(-1) > 0, 'copy').norm_d_()
        return ray

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_extended(ray.x, ray.y) + self.context.z - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent


class CircularSurface(Surface, metaclass=abc.ABCMeta):
    r"""
    Derived class of :py:class:`~Surface` for optical surfaces
    with circular symmetry, i.e. its property
    depends only on the radial distance :math:`r=\sqrt{x^2+y^2}`, in a group of lens.
    Therefore, their surface function can be written as
    :math:`h(x,y)=\hat{h}(x^2+y^2)=\hat{h}(r^2)`.
    Note that :math:`\hat{h}`
    takes as input squared radial distance for computational efficiency purpose.

    :param float radius: Radius of aperture of the surface.
    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param distance: Distance between the surface and the next one.
    :param dict newton_config: Configuration for Newton's method.
        See :ref:`configuration_for_newtons_method` for details.
    """

    def __init__(
        self,
        radius: float,
        material: mt.Material | str,
        distance: Scalar,
        newton_config: dict[str, Any] = None
    ):
        super().__init__(material, distance, newton_config)
        self.radius: float = radius  #: Radius of circular aperture of the surface.

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
        pass

    @property
    @abc.abstractmethod
    def geo_radius(self) -> Ts:
        """
        Geometric radius of the surface, i.e. maximum radial distance that makes
        the surface function mathematically meaningful. A 0D tensor.

        :type: Tensor
        """
        pass

    def h(self, x: Ts, y: Ts) -> Ts:
        return self.h_r2(x.square() + y.square())

    def h_extended(self, x: Ts, y: Ts) -> Ts:
        return self.h_extended_r2(x.square() + y.square())

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
        mask = r2 <= lim2
        return torch.where(mask, phpx, 0), torch.where(mask, phpy, r2)

    def h_extended_r2(self, r2: Ts) -> Ts:
        r"""
        Computes extended version of :math:`\hat{h}(r^2)`.
        See :py:meth:`~h_r2` and :py:meth:`~h_extended`.
        """
        lim2 = self.geo_radius.square()
        return torch.where(r2 <= lim2, self.h_r2(r2), self.h_r2(lim2))

    def sample_rectangular(self, n: Size2d, sampling_curve: Callable[[Ts], Ts]) -> tuple[Ts, Ts]:
        n = size2d(n)
        xy = [torch.linspace(
            -SAMPLE_LIMIT, SAMPLE_LIMIT, _n, device=self.device, dtype=self.dtype
        ) for _n in n]
        if sampling_curve is not None:
            xy = [sampling_curve(x_or_y[_n // 2:]) for x_or_y, _n in zip(xy, n)]
            xy = [torch.cat((-x_or_y.flip([0])[:_n // 2], x_or_y)) for x_or_y, _n in zip(xy, n)]
        x, y = torch.meshgrid(*xy, indexing='xy')  # both 2D
        x, y = x.flatten() * self.radius, y.flatten() * self.radius
        mask = self._valid(x, y)
        return x[mask], y[mask]

    def sample_unipolar(self, n: Size2d) -> tuple[Ts, Ts]:
        n = size2d(n)
        zero = torch.tensor(0., dtype=self.dtype, device=self.device)
        r = torch.linspace(0, self.radius * EDGE_CUTTING, n[0] + 1, device=self.device, dtype=self.dtype)
        r = [r[i].expand(i * n[1]) for i in range(1, n[0] + 1)]  # n[1]*n[0]*(n[0]+1)/2
        r = torch.cat([zero] + r)  # n[1]*n[0]*(n[0]+1)/2+1
        t = [
            torch.arange(i * n[1], device=self.device, dtype=self.dtype) / (n[1] * i) * (2 * torch.pi)
            for i in range(1, n[0] + 1)
        ]
        t = torch.cat([zero] + t)
        return r * t.cos(), r * t.sin()

    def sample_random(self, n: int, sampling_curve: Callable[[Ts], Ts] = None) -> tuple[Ts, Ts]:
        """
        Sample points on the baseline plane randomly.
        This method is subject to change.
        """
        t = torch.rand(n, device=self.device, dtype=self.dtype) * (2 * torch.pi)
        r = torch.rand(n, device=self.device, dtype=self.dtype)
        if sampling_curve is not None:
            r = sampling_curve(r)
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    def sample(self, n: Size2d, mode: SurfSample, **kwargs) -> tuple[Ts, Ts]:
        """
        Sample points on the baseline plane in a regular grid.
        This method is subject to change.
        """
        if mode == 'unipolar':
            return self.sample_unipolar(n)
        elif mode == 'rectangular':
            return self.sample_rectangular(n, **kwargs)
        elif mode == 'random':
            return self.sample_random(n, **kwargs)
        else:
            raise ValueError(f'Unknown sampling mode: {mode}')

    def _f(self, ray: BatchedRay) -> Ts:
        return self.h_extended_r2(ray.r2) + self.context.z - ray.z

    @overload
    def _valid(self, x: Ts, y: Ts) -> Ts:
        pass

    @overload
    def _valid(self, ray: BatchedRay) -> Ts:
        pass

    def _valid(self, *args, **kwargs) -> Ts:
        ba = base.get_bound_args(self._valid, *args, **kwargs)
        if 'ray' in ba.arguments:  # _valid(self, ray: BatchedRay) -> Ts
            ray: BatchedRay = ba.arguments['ray']
            return ray.x.square() + ray.y.square() <= self.radius ** 2
        else:  # _valid(self, x: Ts, y: Ts) -> Ts
            x, y = ba.arguments['x'], ba.arguments['y']
            return x.square() + y.square() <= self.radius ** 2

    def _f_grad(self, ray: BatchedRay) -> Ts:
        phpx, phpy = self.h_grad_extended(ray.x, ray.y, ray.r2)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent


class CircularStop(CircularSurface):
    def intercept(self, ray: BatchedRay) -> BatchedRay:
        raise NotImplementedError()

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def h_r2(self, r2: Ts) -> Ts:
        return torch.zeros_like(r2)

    @property
    def geo_radius(self) -> Ts:
        return torch.tensor(self.radius, device=self.device, dtype=self.dtype)


class SurfaceList(nn.ModuleList, base.TensorContainerMixIn, collections.abc.MutableSequence[CircularSurface]):
    """
    A sequential container of surfaces. This class is derived from
    :py:class:`torch.nn.ModuleList` and implements
    :py:class:`collections.abc.MutableSequence` interface.
    So its instance can be regarded as both a PyTorch module
    and a list of :py:class:`Surface`.

    :param surfaces: A sequence of :py:class:`Surface` objects. Default: ``[]``.
    :type surfaces: Sequence[CircularSurface]
    :param env_material: The material before the first surface.
    :type env_material: :py:class:`~dnois.mt.Material`
    """
    _force_surface: bool = True

    def __init__(
        self,
        surfaces: Sequence[CircularSurface] = None,
        env_material: mt.Material | str = 'vacuum',
    ):
        super().__init__()
        if surfaces is None:
            surfaces = []
        if not isinstance(env_material, mt.Material):
            env_material = mt.get(env_material)

        # This is needed to facilitate MutableSequence operations
        # because torch.nn.ModuleList saves submodules like a dict rather than list
        self._slist: list[CircularSurface] = []
        self._stop_idx = None
        #: Material before the first surface.
        self.env_material: mt.Material = env_material

        self.extend(surfaces)

    def __contains__(self, item) -> bool:
        """:meta private:"""
        return self._slist.__contains__(item)

    def __delitem__(self, key):
        """:meta private:"""
        super().__delitem__(key)
        self._slist.__getitem__(key).context = None
        self._slist.__delitem__(key)

    def __getitem__(self, item):
        """:meta private:"""
        return self._slist.__getitem__(item)

    def __iadd__(self, other: Sequence[CircularSurface]) -> Self:
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
        return SurfaceList(list(self._slist.__reversed__()), self.env_material)

    def __setitem__(self, key: int, value: CircularSurface):
        """:meta private:"""
        self._welcome(value)
        super().__setitem__(key, value)
        self._slist.__setitem__(key, value)

    def __add__(self, other: Sequence[CircularSurface]) -> 'SurfaceList':
        """:meta private:"""
        return SurfaceList(self._slist + list(other), self.env_material)

    def __repr__(self) -> str:
        """:meta private:"""
        _repr = super().__repr__()[:-1]  # remove that last parentheses
        _repr += f'  env_material={repr(self.env_material)}\n)'
        return _repr

    def __dir__(self):
        """:meta private:"""
        return super().__dir__() + ['env_material']

    def append(self, surface: CircularSurface):
        """:meta private:"""
        self._welcome(surface)
        super().append(surface)

    def clear(self):
        """:meta private:"""
        for s in self._slist:
            s.context = None
        self._slist.clear()
        self._super_clear()

    def count(self, value: CircularSurface) -> int:
        """:meta private:"""
        return self._slist.count(value)

    def extend(self, surfaces: Sequence[CircularSurface]):
        """:meta private:"""
        self._welcome(*surfaces)
        super().extend(surfaces)

    def index(self, value: CircularSurface, start: int = 0, stop: int = ...) -> int:
        """:meta private:"""
        if stop is ...:
            return self._slist.index(value, start)
        else:
            return self._slist.index(value, start, stop)

    def insert(self, index: int, surface: CircularSurface):
        """:meta private:"""
        self._welcome(surface)
        super().insert(index, surface)
        self._slist.insert(index, surface)

    def pop(self, index: int = -1) -> CircularSurface:
        """:meta private:"""
        s = super().pop(index)
        s.context = None
        return s

    def remove(self, value: CircularSurface):
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
        if name.isdigit() and isinstance(module, CircularSurface):
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

    @property
    def surfaces(self) -> list[CircularSurface]:
        """
        Returns a list of contained surfaces.

        :type: list[Surface]
        """
        return [s for s in self._slist]

    @property
    def length(self) -> Ts | None:
        """
        Returns the distance between baselines of the first and the last surfaces
        as a 0D tensor. If there is no more than one surface, returns ``None``.

        :type: Tensor or None
        """
        if len(self._slist) < 2:
            return None
        return sum(s.distance for s in self._slist[:-1])

    @property
    def total_length(self) -> Ts | None:
        """
        Returns the sum of :py:attr:`Surface.distance` of all the surfaces
        as a 0D tensor. If there is no surface, returns ``None``.

        :type: Tensor or None
        """
        if len(self._slist) == 0:
            return None
        return sum(s.distance for s in self._slist)

    @property
    def stop_idx(self) -> int:
        """
        Index of the aperture stop. Returns ``None`` if no stop is found.

        :type: int
        """
        return self._stop_idx

    @property
    def stop(self) -> CircularSurface:
        """
        The aperture stop object. Returns ``None`` if no stop is found.
        Note that it need not return an instance of :py:class:`CircularStop`.

        :type: :py:class:`CircularStop`
        """
        idx = self._stop_idx
        return None if idx is None else self._slist[idx]

    def _welcome(self, *new: CircularSurface):
        for surface in new:
            if self._force_surface and not isinstance(surface, CircularSurface):
                msg = f'An instance of {CircularSurface.__name__} expected, got {type(surface)}'
                raise TypeError(msg)
            if surface.context is not None:
                raise RuntimeError(f'A surface already contained in a lens group '
                                   f'cannot be inserted to another one')
            surface.context = Context(surface, self)

        for s1 in self._slist:
            for s2 in new:
                if id(s1) == id(s2):
                    raise ValueError('Trying to add a surface into a lens group containing it')

    def _super_clear(self):
        for idx in range(len(self) - 1, -1, -1):
            super().__delitem__(idx)
