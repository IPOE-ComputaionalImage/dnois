import copy
import warnings

import torch

from ... import base
from ...base.typing import Ts, Literal, Self

__all__ = [
    'BatchedRay',
    'NoValidRayError',
]


def _get_dd(**tensors: Ts):
    tensors = {k: v for k, v in tensors.items() if not (v is None or isinstance(v, float))}
    if len(tensors) == 0:
        return
    for k, ts in tensors.items():
        if not torch.is_tensor(ts):
            raise TypeError(f'torch.Tensor expected for {k}, got {type(ts).__name__}')
    first_key = next(iter(tensors))
    first = tensors[first_key]
    device, dtype = first.device, first.dtype
    for k, ts in tensors.items():
        if ts.device != device or ts.dtype != dtype:
            raise RuntimeError(f'Inconsistent device and dtype detected: ({device}, {dtype}) '
                               f'for {first_key} but ({ts.device}, {ts.dtype}) for {k}')
    return device, dtype


def _2tensor(x: float | Ts | None, device, dtype) -> Ts | None:
    if x is None or torch.is_tensor(x):
        return x
    else:  # float
        return torch.tensor(x, device=device, dtype=dtype)


def _check_shape_compatible(*tensors: Ts):
    shapes = [ts.shape for ts in tensors if ts is not None]
    torch.broadcast_shapes(*shapes)


class NoValidRayError(RuntimeError):
    """Error meaning that valid rays vanish."""
    pass


# WARNING: do not modify the tensors in this class in an in-place manner
# TODO: deal with floating-point error
class BatchedRay(base.TensorContainerMixIn):
    r"""
    A class representing a batch of rays, which means both the origin and direction are
    tensors with the last dimension as 3, representing three coordinates x, y and z.
    Shapes of all the tensor arguments of ``__init__`` method and
    all the tensors associated to an object of this class can be different but must
    be broadcastable, other than the last dimension of ``origin`` and ``direction``.
    The broadcast shape is called *shape of rays*.

    One may intend to record the accumulative optical path lengths (OPL) and phases or rays.
    In this case, provide initial OPLs or phases as ``init_opl`` or ``init_phase``, respectively.
    If there is no initial value, just provide ``0.``.
    If either of them is not provided, corresponding quantity will not be recorded.

    .. note::

        As mentioned above, tensors bound to a ray object may have different shapes,
        so do its properties of tensor type. However, their shapes should be broadcastable,
        and :py:attr:`.shape` always returns broadcast shape.

    .. warning::

        Some properties (attributes) depending on other properties may be cached for
        efficiency purpose. DO NOT modify the tensor properties with in-place operations.

    :param Tensor origin: The origin of the rays. A tensor of shape ``(..., 3)``.
    :param Tensor direction: The direction of the rays. A tensor of shape ``(..., 3)``.
    :param wl: The wavelength of the rays. A tensor of shape ``(...)``.
    :type wl: float or Tensor
    :param init_opl: Initial optical path lengths. A tensor of shape ``(...)``.
        Default: do not record optical path lengths.
    :type init_opl: float or Tensor
    :param init_phase: Initial phase. A tensor of shape ``(...)``.
        Default: do not record phase.
    :type init_phase: float or Tensor
    """

    __slots__ = (
        '_d_normed',
        '_o_modified',
        '_ts',
    )

    def __init__(
        self,
        origin: Ts,
        direction: Ts,
        wl: float | Ts,
        init_opl: float | Ts = None,
        init_phase: float | Ts = None,
    ):
        if origin.size(-1) != 3 or direction.size(-1) != 3:
            raise ValueError(
                f'The last dimension of origin and direction must be 3, '
                f'got {origin.size(-1)} and {direction.size(-1)}'
            )

        device, dtype = _get_dd(
            origin=origin, direction=direction, wl=wl, init_opl=init_opl, init_phase=init_phase
        )
        wl = _2tensor(wl, device, dtype)
        init_opl = _2tensor(init_opl, device, dtype)
        init_phase = _2tensor(init_phase, device, dtype)
        _check_shape_compatible(origin[..., 0], direction[..., 0], wl, init_opl, init_phase)
        self._ts: dict[str, Ts] = {
            'o': origin,
            'd': direction,
            'wl': wl,
            'v': torch.tensor(True, dtype=torch.bool, device=device),
        }
        self._d_normed = False
        self._o_modified = False

        if init_opl is not None:
            self._ts['opl'] = init_opl
        if init_phase is not None:
            self._ts['ph'] = init_phase

    def __repr__(self):
        shape, recording_opl, recording_phase = self.shape, self.recording_opl, self.recording_phase
        return f'BatchedRay({shape=}, {recording_opl=}, {recording_phase=})'

    def broadcast_(self) -> Self:
        """
        Broadcast all the associated tensors to shape of rays :py:attr:`.shape`.

        :return: self
        """
        shape = self.shape
        for k in list(self._ts.keys()):
            self._ts[k] = torch.broadcast_to(self._ts[k], (shape + (3,)) if k in ('o', 'd') else shape)
        return self

    def clone(self, deep: bool = False) -> 'BatchedRay':
        """
        Return a copy of this object.

        :param bool deep: Whether to clone associated tensors by calling
            :py:meth:`torch.Tensor.clone`. If ``False``, returned object
            will hold the reference to the tensors bound to this object.
        :return: The cloned object.
        :rtype: :py:class:`BatchedRay`
        """
        new_ray = copy.copy(self)
        new_ray._ts = new_ray._ts.copy()
        if deep:
            for k in list(new_ray._ts):
                new_ray._ts[k] = new_ray._ts[k].clone()
        return new_ray

    def discard_(self) -> Ts:
        """
        Discard invalid rays. This method can be called only if the shape
        of rays (i.e. ``len(self.shape)``) is 1. :py:meth:`.flatten_`
        can be called to make an instance to satisfy this requirement.
        Note that this will lead to change of :py:attr:`.shape` and data copy.

        :return: A 1D :py:class:`torch.LongTensor` indicating the indices
            of remaining rays.
        :rtype: torch.LongTensor
        """
        if len(self.shape) != 1:
            raise RuntimeError(f'{self.discard_.__name__} can only be called on 1D array of rays')
        valid = torch.argwhere(self.valid)  # N_valid x 1
        valid = valid.squeeze()  # N_valid
        if valid.size(0) == self._ts['v'].size(0):  # all valid
            return valid

        def _discard(ts: Ts) -> Ts:
            if ts.ndim == 1 or ts.size(-2) == 1:
                ts = torch.broadcast_to(ts, (self.shape[0], 3))
            return ts[valid]

        self._ts['v'] = self._ts['v'][valid]
        self._ts['o'] = _discard(self._ts['o'])  # N_valid x 3
        self._ts['d'] = _discard(self._ts['d'])  # N_valid x 3
        self._ts['wl'] = self._ts['wl'][valid]  # N_valid
        if self.recording_opl:
            self._ts['opl'] = self._ts['opl'][valid]  # N_valid
        if self.recording_phase:
            self._ts['ph'] = self._ts['ph'][valid]
        return valid

    def flatten_(self) -> Self:
        """
        Flatten the shape of rays to 1D.

        :return: self
        """
        shape = self.shape
        self._ts['v'] = torch.flatten(self.valid.broadcast_to(shape))
        self._ts['o'] = torch.flatten(self.o.broadcast_to(shape + (3,)), 0, -2)
        self._ts['d'] = torch.flatten(self.d.broadcast_to(shape + (3,)), 0, -2)
        self._ts['wl'] = torch.flatten(self.wl.broadcast_to(shape))
        if self.recording_opl:
            self._ts['opl'] = torch.flatten(self.opl.broadcast_to(shape))
        if self.recording_phase:
            self._ts['ph'] = torch.flatten(self.phase.broadcast_to(shape))
        return self

    def copy_valid_(self) -> Self:
        """
        Replace all invalid rays with one of valid ray.
        There is little data copy in this method.

        :return: self
        """
        v = self._ts['v']
        if v.all():
            return self
        if not v.any():
            raise NoValidRayError(f'There is no valid ray')

        v = v.broadcast_to(self.shape)
        idx = v.nonzero(as_tuple=False)[0]  # The indices of the first valid element
        idx = idx.tolist()
        o, d = self._ts['o'].broadcast_to(v.shape + (3,)), self._ts['d'].broadcast_to(v.shape + (3,))
        self.o = torch.where(v.unsqueeze(-1), o, o[*idx].clone())
        self.d = torch.where(v.unsqueeze(-1), d, d[*idx].clone())
        wl = self._ts['wl'].broadcast_to(v.shape)
        self.wl = torch.where(v, wl, wl[*idx].clone())
        if self.recording_opl:
            opl = self._ts['opl'].broadcast_to(v.shape)
            self.opl = torch.where(v, opl, opl[*idx].clone())
        if self.recording_phase:
            phase = self._ts['ph'].broadcast_to(v.shape)
            self.opl = torch.where(v, phase, phase[*idx].clone())
        return self

    def march(self, t: Ts, n: float | Ts = None) -> 'BatchedRay':
        """Out-of-place version of :py:meth:`.march_`."""
        return self.clone().march_(t, n)

    def march_(self, t: Ts, n: float | Ts = None) -> Self:
        """
        March forward by a distance ``t``. The origin will be updated.

        :param Tensor t: Propagation distance. A tensor whose shape must be broadcastable
            with the shape of rays.
        :param n: Refractive index of the medium in where the rays propagate.
            A float or a tensor with shape of rays. Default: 1.
        :type n: float or Tensor
        :return: self
        """
        self.o = self._ts['o'] + self.d_norm * t.unsqueeze(-1)
        _opl, _ph = self.recording_opl, self.recording_phase
        if _opl or _ph:
            if n is None:
                warnings.warn('Refractive index not specified for coherent rays')
            opl = t if n is None else n * t
            if _opl:
                self.opl = self.opl + opl
            if _ph:
                self.phase = self.phase + 2 * torch.pi * self.wl * opl
        return self

    def march_to(self, z: Ts, n: float | Ts = None) -> 'BatchedRay':
        """Out-of-place version of :py:meth:`.march_to_`."""
        return self.clone().march_to_(z, n)

    def march_to_(self, z: Ts, n: float | Ts = None) -> Self:
        """
        March forward to a given z value ``z``. This is similar to :py:meth:`.march_`,
        but all the rays end up with identical z coordinate rather than marching distance.

        :param Tensor z: Target value of z coordinate.
            A tensor whose shape must be broadcastable with the shape of rays.
        :param n: Refractive index of the medium in where the rays propagate.
            A float or a tensor with shape of rays. Default: 1.
        :type n: float or Tensor
        :return: self
        """
        t = (z - self.z) / self.d_z
        return self.march_(t, n)

    def norm_d_(self) -> Self:
        """
        Normalize the direction in place and return ``self``.

        :return: self
        """
        if self._d_normed:
            return self
        d = self._ts['d']
        self._ts['d'] = d / d.norm(2, -1, True)
        self._d_normed = True
        return self

    def to_(self, *args, **kwargs) -> Self:
        """
        Call :py:meth:`torch.Tensor.to` for all the tensors bound to ``self``
        and update them by the returned tensor.

        This method will not update the dtype of ``self.valid``, which is always
        ``torch.bool``.

        :param args: Positional arguments accepted by :py:meth:`torch.Tensor.to`.
        :param kwargs: Keyword arguments accepted by :py:meth:`torch.Tensor.to`.
        :return: self
        """
        for k in list(self._ts.keys()):
            nt = self._ts[k].to(*args, **kwargs)
            if k == 'v':
                nt = nt.to(torch.bool)
            self._ts[k] = nt
        return self

    def update_valid_(self, valid: Ts, action: Literal['discard', 'copy'] = None) -> Self:
        """
        Update the validity flags with ``valid``. The new validity flag will be
        its logical ``&`` with the old.

        :param Tensor valid: Another validity flag.
        :param str action: The action to take after updating validity flags.
            Call :py:meth:`.discard_` if ``'discard'``, or :py:meth:`.copy_valid_`
            if ``'copy'``, or nothing if ``None``. Default: ``None``.
        :return: self
        """
        if valid.dtype != torch.bool:
            raise TypeError('Only bool tensors are supported.')
        self.valid = torch.logical_and(self._ts['v'], valid)
        if action == 'discard':
            self.discard_()
        elif action == 'copy':
            self.copy_valid_()
        return self

    @property
    def shape(self) -> torch.Size:
        """
        Shape of rays, i.e. the broadcast shape of all the tensors bound to ``self``.

        :type: torch.Size
        """
        shapes = [ts.shape[:-1] if k in ('o', 'd') else ts.shape for k, ts in self._ts.items()]
        return torch.broadcast_shapes(*shapes)

    @property
    def o(self) -> Ts:
        """
        The origin of the rays.

        :type: Tensor
        """
        return self._ts['o']

    @o.setter
    def o(self, value: Ts):
        if value.ndim < 1:
            raise base.ShapeError(f'Non-scalar tensor expected, got shape ({value.shape})')
        self._check_shape(value.shape[:-1], 'origin')
        self._ts['o'] = value.to(self._ts['o'])
        self._o_modified = True

    @property
    def d(self) -> Ts:
        """
        The direction of the rays. The length or returned direction
        (i.e. 2-norm along the last dimension) may not be normalized to 1.
        Use :py:meth:`norm_d_` to normalize the direction or
        :py:attr:`d_norm` to get a normalized copy of the direction.

        :type: Tensor
        """
        return self._ts['d']

    @d.setter
    def d(self, value: Ts):
        if value.ndim < 1:
            raise base.ShapeError(f'Non-scalar tensor expected, got shape ({value.shape})')
        self._check_shape(value.shape[:-1], 'direction')
        self._ts['d'] = value.to(self._ts['d'])
        self._d_normed = False

    @property
    def d_norm(self) -> Ts:
        """
        The normalized direction of the rays. See :py:attr:`.d`
        for the difference between these two properties.

        :type: Tensor
        """
        d = self._ts['d']
        return d if self._d_normed else d / d.norm(2, -1, True)

    @property
    def wl(self) -> Ts:
        """
        The wavelengths of rays.

        :type: Tensor
        """
        return self._ts['wl']

    @wl.setter
    def wl(self, value: Ts):
        self._check_shape(value.shape, 'wavelength')
        self._ts['wl'] = value.to(self._ts['wl'])

    @property
    def valid(self) -> Ts:
        """
        The validity flag of the rays.

        :type: :py:class:`torch.BoolTensor`
        """
        return self._ts['v']

    @valid.setter
    def valid(self, value: Ts):
        if value.dtype != torch.bool:
            raise TypeError('Rays\' validity flag must be a bool tensor.')
        self._check_shape(value.shape, 'validity')
        self._ts['v'] = value.to(self._ts['v'])

    @property
    def opl(self) -> Ts | None:
        """
        The accumulative OPL of the rays in meters which is always positive.
        This property is ``None`` if OPL is not recorded.

        .. warning::
            Direct modification to this property do not reflect on :py:attr:`.phase`
            if it is recorded. Using :py:meth:`march_` to ensure they are updated synchronously.

        :type: Tensor
        """
        return self._ts.get('opl', None)

    @opl.setter
    def opl(self, value: Ts):
        self._check_shape(value.shape, 'OPL')
        if value.lt(0).any():
            raise ValueError('OPL must be non-negative.')
        self._ts['opl'] = value.to(self._ts['opl'])

    @property
    def phase(self) -> Ts | None:
        r"""
        The phase shift of the rays, ranging in :math:`[0,2\pi]`.
        This property is ``None`` if phase is not recorded.

        .. warning::
            Direct modification to this property do not reflect on :py:attr:`.opl`
            if it is recorded. Using :py:meth:`march_` to ensure they are updated synchronously.

        :type: Tensor
        """
        return self._ts.get('ph', None)

    @phase.setter
    def phase(self, value: Ts):
        self._check_shape(value.shape, 'phase')
        value = value.to(self._ts['ph'])
        if value.lt(0).any() or value.gt(2 * torch.pi).any():
            value = value.remainder(2 * torch.pi)
        self._ts['ph'] = value

    @property
    def recording_opl(self) -> bool:
        """
        Whether OPLs are recorded.

        :type: bool
        """
        return 'opl' in self._ts

    @property
    def recording_phase(self) -> bool:
        """
        Whether phases are recorded.

        :type: bool
        """
        return 'ph' in self._ts

    @property
    def x(self) -> Ts:
        """
        :math:`x` coordinate of the origin. Its shape is that of :py:attr:`.o`
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 0]

    @property
    def y(self) -> Ts:
        """
        :math:`y` coordinate of the origin. Its shape is that of :py:attr:`.o`
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 1]

    @property
    def z(self) -> Ts:
        """
        :math:`z` coordinate of the origin. Its shape is that of :py:attr:`.o`
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 2]

    @property
    def r2(self) -> Ts:
        """
        :math:`x^2+y^2` of the origin. Its shape is that of :py:attr:`.o`
        without the last dimension.

        :type: Tensor
        """
        if not self._o_modified and 'r2' in self._ts:
            return self._ts['r2']
        r2 = self.x.square() + self.y.square()
        self._ts['r2'] = r2
        return r2

    @property
    def d_x(self) -> Ts:
        """
        The projection of the direction to :math:`x` axis.
        Its shape is that of :py:attr:`.d` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 0]

    @property
    def d_y(self) -> Ts:
        """
        The projection of the direction to :math:`y` axis.
        Its shape is that of :py:attr:`.d` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 1]

    @property
    def d_z(self) -> Ts:
        """
        The projection of the direction to :math:`z` axis.
        Its shape is that of :py:attr:`.d` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 2]

    def _check_shape(self, shape: tuple[int, ...], name: str):
        if not base.broadcastable(shape, self.shape):
            raise base.ShapeError(f'Trying to assign a tensor with incompatible shape {shape} '
                                  f'to {name}, while shape of rays is {self.shape}')

    def _delegate(self):
        return self._ts['o']
