import abc
import collections.abc

import torch
from torch import nn

from dnois import mt
from dnois.base.typing import Sequence, Ts, Any, Callable, Literal, Scalar, scalar

from .ray import BatchedRay, NoValidRayError

__all__ = [
    'NT_EPSILON',
    'NT_MAX_ITERATION',
    'NT_THRESHOLD',
    'NT_THRESHOLD_STRICT',
    'NT_UPDATE_BOUND',

    'Context',
    'SurfaceList',
    'Surface',
]

NT_MAX_ITERATION: int = 10
NT_THRESHOLD: float = 20e-9
NT_THRESHOLD_STRICT: float = 20e-9
NT_UPDATE_BOUND: float = 5.
NT_EPSILON: float = 1e-9

SAMPLE_LIMIT: float = 0.98


class Context:
    """
    A class representing the context of a :py:class:`~Surface` in a lens group. As component of the
    lens group, a surface does not hold the reference to the group but can access
    the information that depends on other surfaces in it via this class.
    Every surface contained in a lens group has a related context object.
    If it is not contained in any group, its context attribute is ``None``.

    :param Surface surface: The host surface that this context belongs to.
    :param SurfaceList lens_group: The lens group containing the surface.
    """

    def __init__(self, surface: 'Surface', lens_group: 'SurfaceList'):
        self.surface: 'Surface' = surface  #: The host surface that this context belongs to.
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


class Surface(nn.Module, metaclass=abc.ABCMeta):
    r"""
    Base class for optical surfaces with circular apertures in a lens group.
    The position of a surface in a lens group is specified by a single z-coordinate,
    which is called its *baseline*. Baseline of the first surface is 0.

    The geometric shape of a surface is described by an equation
    :math:`z=h(x,y;\mathbf{\theta})+z_\text{baseline}`, which has different form
    for each surface types. The function :math:`h`, called *surface function*,
    is a 2D function of lateral coordinates :math:`(x,y)` which depends on
    parameters of the surface :math:`\mathbf{\theta`} and satisfies :math:`h(0,0)=0`.

    .. note::

        All the quantities with length dimension is represented in meters.
        They include value of coordinates, optical path length and wavelength, etc.

    :param float radius: Radius of aperture of the surface.
    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param float distance: Distance between the surface and the next one.
    :param dict newton_config: Configuration for Newton's method. TODO
    """

    def __init__(
        self,
        radius: float,
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
        self.radius: float = radius  #: Radius of circular aperture of the surface.
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
    def h(self, x: Ts, y: Ts = None) -> Ts:  # TODO: use typing.overload
        r"""
        Compute surface function :math:`h(x,y;\mathbf{\theta})`.

        :param x:
        :param y:
        :return:
        """
        pass

    @abc.abstractmethod
    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Compute the partial derivatives of surface function
        :math:`\pfrac{h(x,y;\mathbf{\theta})}{x}` and :math:`\pfrac{h(x,y;\mathbf{\theta})}{y}`.

        :param x:
        :param y:
        :param r2:
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def geo_radius(self) -> Ts:
        pass

    def h_extended(self, x: Ts, y: Ts = None) -> Ts:
        r2 = x if y is None else x.square() + y.square()
        lim2 = self.geo_radius.square()
        return torch.where(r2 <= lim2, self.h(r2), self.h(lim2))

    def h_grad_extended(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        lim2 = self.geo_radius.square()
        phpx, phpy = self.h_grad(x, y, r2)
        mask = r2 <= lim2
        return torch.where(mask, phpx, 0), torch.where(mask, phpy, r2)

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        magnitude = self._nt_threshold_strict / torch.finfo(ray.o.dtype).eps
        if torch.abs(ray.z - self.context.z).gt(magnitude).any():
            new_z = self.context.z - magnitude if forward else self.context.z + magnitude
            ray.march_to_(new_z, self.context.material_before.n(ray.wl, 'm'))

        if forward:
            valid = torch.logical_and(self._eval_f(ray) >= 0, ray.d_z > 0)
        else:
            valid = torch.logical_and(self._eval_f(ray) <= 0, ray.d_z < 0)
        ray.update_valid_(valid, 'copy')

        return self.refract(self.intersect(ray), forward)

    def intersect(self, ray: BatchedRay) -> BatchedRay:
        t = (self.context.z - ray.z) / ray.d_z
        cnt = 0  # equal to numbers of derivative computation
        new_ray = ray.clone(False)
        new_ray.norm_d_()
        with torch.no_grad():
            while True:
                new_ray.o = ray.o + new_ray.d_norm * t.unsqueeze(-1)  # do not compute opl for root finder
                r2 = new_ray.x.square() + new_ray.y.square()
                f_value = self._eval_f(new_ray, r2)
                if torch.all(f_value.abs() < self._nt_threshold) or cnt >= self._nt_max_iteration:
                    break

                t = t - self._newton_descent(new_ray, f_value, r2)
                cnt += 1

        # the second argument cannot be replaced by f_value because of computational graph
        r2 = new_ray.x.square() + new_ray.y.square()
        t = t - self._newton_descent(new_ray, self._eval_f(new_ray, r2), r2)
        ray = ray.march(t, self.context.material_before.n(ray.wl, 'm'))
        ray.update_valid_(
            self._valid(ray.x, ray.y) &
            (self._eval_f(ray).abs() < self._nt_threshold_strict) &
            (t > 0), 'copy'
        )
        return ray

    def normal(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        """
        Point to positive z. unit vector. TODO

        :param x:
        :param y:
        :param r2:
        :return:
        """
        phpx, phpy = self.h_grad(x, y, r2)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        ray = ray.clone(False)
        n = self.normal(ray.x, ray.y)
        if not forward:
            n = -n
        i = ray.d_norm
        if forward:
            miu = self.context.material_before.n(ray.wl, 'm') / self.material.n(ray.wl, 'm')
        else:
            miu = self.material.n(ray.wl, 'm') / self.context.material_before.n(ray.wl, 'm')
        miu = miu.unsqueeze(-1)
        ni = torch.sum(n * i, dim=-1).unsqueeze(-1)
        nt2 = 1 - miu.square() * (1 - ni.square())
        t = torch.sqrt(nt2.relu_()) * n + miu * (i - ni * n)
        ray.d = t
        ray.update_valid_(nt2.squeeze(-1) > 0, 'copy').norm_d_()
        return ray

    def sample(
        self, n: tuple[int, int],
        mode: Literal['rectangular', 'circular'],
        sampling_curve: Callable[[Ts], Ts] = None,
        **kwargs
    ) -> tuple[Ts, Ts]:
        if mode == 'circular':
            t = torch.arange(n[0], **kwargs).unsqueeze(0) / n[0] * (2 * torch.pi)
            r = torch.linspace(1 / n[1], SAMPLE_LIMIT, n[1], **kwargs).unsqueeze(1)
            if sampling_curve is not None:
                r = sampling_curve(r)
            r = r * self.radius
            return torch.flatten(r * t.cos()), torch.flatten(r * t.sin())
        elif mode == 'rectangular':  # TODO: this seems meaningless
            xy = [torch.linspace(-SAMPLE_LIMIT, SAMPLE_LIMIT, _n, **kwargs) for _n in n]
            if sampling_curve is not None:
                xy = [sampling_curve(x_or_y[_n // 2:]) for x_or_y, _n in zip(xy, n)]
                xy = [torch.cat((-x_or_y.flip([0])[:_n // 2], x_or_y)) for x_or_y, _n in zip(xy, n)]
            x, y = torch.meshgrid(*xy, indexing='xy')
            return x.flatten() * self.radius, y.flatten() * self.radius
        else:
            raise ValueError(f'Unknown mode: {mode}')

    def sample_random(self, n: int, sampling_curve: Callable[[Ts], Ts] = None, **kwargs) -> tuple[Ts, Ts]:
        t = torch.rand(n, **kwargs) * (2 * torch.pi)
        r = torch.rand(n, **kwargs)
        if sampling_curve is not None:
            r = sampling_curve(r)
        r = r * self.radius
        return r * t.cos(), r * t.sin()

    @property
    def device(self) -> torch.device:
        return self.distance.device

    def _eval_f(self, ray: BatchedRay, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = ray.x.square() + ray.y.square()
        return self.h_extended(r2) + self.context.z - ray.z

    def _eval_f_grad(self, ray: BatchedRay, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = ray.x.square() + ray.y.square()
        phpx, phpy = self.h_grad_extended(ray.x, ray.y, r2)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _valid(self, x: Ts, y: Ts) -> Ts:
        return x.square() + y.square() < self.radius ** 2

    def _newton_descent(self, ray: BatchedRay, f_value: Ts, r2: Ts = None) -> Ts:
        derivative_value = torch.sum(ray.d_norm * self._eval_f_grad(ray, r2), dim=-1)
        descent = f_value / (derivative_value + self._nt_epsilon)
        descent = torch.clip(descent, -self._nt_update_bound, self._nt_update_bound)
        return descent


class SurfaceList(nn.Module, collections.abc.MutableSequence):
    _force_surface: bool = True

    def __init__(
        self,
        surfaces: Sequence[Surface] = None,
        env_material: mt.Material | str = 'vacuum',
    ):
        super().__init__()
        if surfaces is None:
            surfaces = []
        if not isinstance(env_material, mt.Material):
            env_material = mt.get(env_material)

        # This is needed to allow the surfaces to be recognized as submodules of self
        self._surfaces = nn.ModuleList([])
        self._slist: list[Surface] = []
        #: Material before the first surface.
        self.env_material: mt.Material = env_material

        self.extend(surfaces)

    def __contains__(self, item):
        return self._slist.__contains__(item)

    def __delitem__(self, key):
        self._slist.__getitem__(key).context = None
        self._slist.__delitem__(key)
        self._surfaces.__delitem__(key)

    def __getitem__(self, item):
        return self._slist.__getitem__(item)

    def __iadd__(self, other):
        raise RuntimeError(f'+= for LensGroup is prohibited. Use extend instead')

    def __iter__(self):
        return self._slist.__iter__()

    def __len__(self):
        return self._slist.__len__()

    def __reversed__(self):
        return self._slist.__reversed__()

    def __setitem__(self, key, value):
        self._welcome(value)
        self._slist.__setitem__(key, value)
        self._surfaces.__setitem__(key, value)

    def append(self, surface: Surface):
        self._welcome(surface)
        self._slist.append(surface)
        self._surfaces.append(surface)

    def clear(self):
        for s in self._slist:
            s.context = None
        self._slist.clear()
        self._surfaces = nn.ModuleList([])

    def count(self, value) -> int:
        return self._slist.count(value)

    def extend(self, surfaces: Sequence[Surface]):
        self._welcome(*surfaces)
        self._slist.extend(surfaces)
        self._surfaces.extend(surfaces)

    def index(self, value, start: int = 0, stop: int = ...) -> int:
        if stop is ...:
            return self._slist.index(value, start)
        else:
            return self._slist.index(value, start, stop)

    def insert(self, index: int, surface: Surface):
        self._welcome(surface)
        self._slist.insert(index, surface)
        self._surfaces.insert(index, surface)

    def pop(self, index=-1) -> Surface:
        s = self._slist.pop(index)
        s.context = None
        self._surfaces.__delitem__(index)
        return s

    def remove(self, value):
        idx = self.index(value)
        self.pop(idx)

    def reverse(self):
        self._slist.reverse()
        self._surfaces = nn.ModuleList(self._slist)

    def forward(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        for s in (self._slist if forward else reversed(self._slist)):
            try:
                ray = s(ray, forward)
            except NoValidRayError as e:
                idx = self.index(s)
                e.add_note(f'Valid rays vanish during the forward pass of surface {idx}')
                raise e
        return ray

    @property
    def surfaces(self) -> list[Surface]:
        return [s for s in self._slist]

    @property
    def length(self) -> Ts | None:
        if len(self._slist) < 2:
            return None
        return sum(s.distance for s in self._slist[:-1])

    @property
    def total_length(self) -> Ts | None:
        if len(self._slist) == 0:
            return None
        return sum(s.distance for s in self._slist)

    def _welcome(self, *new: Surface):
        for surface in new:
            if self._force_surface and not isinstance(surface, Surface):
                raise TypeError(f'An instance of Surface expected, got {type(surface)}')
            if surface.context is not None:
                raise RuntimeError(f'A surface already contained in a lens group'
                                   f'cannot be inserted to another one')
            surface.context = Context(surface, self)

        for s1 in self._slist:
            for s2 in new:
                if id(s1) == id(s2):
                    raise ValueError('Trying to add a surface into a lens group containing it')
