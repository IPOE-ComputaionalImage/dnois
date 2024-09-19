import warnings

import torch

from dnois.base.typing import Ts, Literal

__all__ = [
    'BatchedRay',
]


class BatchedRay:
    r"""
    A class representing a batch of rays, which means both the origin and direction is
    a tensor of shape ``(..., 3)``. The last dimension represents the three
    coordinates x, y and z. The leading dimensions ``...`` for origin
    and direction can be different but must be broadcastable.
    The broadcast shape without the last dimension is called *shape of rays*.

    :param Ts origin: The origin of the rays. A tensor of shape ``(..., 3)``.
    :param Ts direction: The direction of the rays. A tensor of shape ``(..., 3)``.
    :param Ts wl: The wavelength of the rays. A tensor of shape ``(...)``.
    :param bool coherent: Whether the rays are coherent. Default: ``False``.
    """

    __slots__ = (
        '__weakref__',
        '_coherent',
        '_d_normed',
        '_shape',
        '_ts',
    )

    def __init__(self, origin: Ts, direction: Ts, wl: Ts, coherent: bool = False):
        if origin.dtype != direction.dtype:
            raise ValueError(
                f'The dtype of origin and direction must be the same, '
                f'got {origin.dtype} and {direction.dtype}'
            )
        if origin.size(-1) != 3 or direction.size(-1) != 3:
            raise ValueError(
                f'The last dimension of origin and direction must be 3, '
                f'got {origin.size(-1)} and {direction.size(-1)}'
            )

        _shape = torch.broadcast_shapes(origin.shape, direction.shape)[:-1]
        self._ts = {
            'o': origin,
            'd': direction,
            'wl': torch.broadcast_to(wl, _shape),
            'v': torch.ones(_shape, dtype=torch.bool),
        }
        self._d_normed = False
        self._coherent = coherent  #: Whether this ray is coherent

        if coherent:
            self._ts['opl'] = torch.zeros(_shape, dtype=origin.dtype)

    def __repr__(self):
        return f'BatchedRay(shape={self.shape}, coherent={self._coherent})'

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
        valid = self._ts['v']
        if valid != 1:
            raise RuntimeError('Only the invalid rays in 1D array of rays can be discarded')
        valid = torch.argwhere(valid)  # N_valid x 1
        valid = valid.squeeze()  # N_valid
        if valid.size(0) == self._ts['v'].size(0):  # all valid
            return valid

        def _discard(ts: Ts) -> Ts:
            if ts.ndim == 1 or ts.size(-2) == 1:
                ts = torch.broadcast_to(ts, (self._shape[0], 3))
            return ts[valid]

        self._ts['o'] = _discard(self._ts['o'])  # N_valid x 3
        self._ts['d'] = _discard(self._ts['d'])  # N_valid x 3
        self._ts['wl'] = self._ts['wl'][valid]
        self._ts['v'] = torch.ones(valid.size(0), device=valid.device, dtype=torch.bool)
        if self._coherent:
            self._ts['opl'] = self._ts['opl'][valid]  # N_valid
        return valid

    def flatten_(self):
        """
        Flatten the shape of rays to 1D.

        :return: ``None``
        """
        v = self._ts['v']
        o = torch.broadcast_to(self._ts['o'], v.shape + (3,))
        d = torch.broadcast_to(self._ts['d'], v.shape + (3,))
        self._ts.update(
            o=torch.flatten(o, 0, -2),
            d=torch.flatten(d, 0, -2),
            v=torch.flatten(v),
            wl=torch.flatten(self._ts['wl'])
        )
        if self._coherent:
            self._ts['opl'] = torch.flatten(self._ts['opl'])

    def copy_valid_(self):
        """
        Replace all invalid rays with one of valid ray.
        There is little data copy in this method.

        :return: ``None``
        """
        v = self._ts['v']
        if v.all():
            return

        idx = -1
        vv = v.view(-1)
        for i in range(v.numel()):  # find a valid ray
            if vv[i]:
                idx = i
        if idx == -1:
            raise RuntimeError(f'There is no valid ray')
        o, d, wl = self._ts['o'], self._ts['v'], self._ts['wl']
        o[v.logical_not()] = o.view(-1, 3)[idx]  # shape: (3,)
        d[v.logical_not()] = d.view(-1, 3)[idx]
        wl[v.logical_not()] = wl.view(-1)[idx]
        if self._coherent:
            opl = self._ts['opl']
            opl[v.logical_not()] = opl.view(-1)[idx]

    def norm_d_(self):
        """
        Normalize the direction in place and return ``self``.

        :return: ``None``
        """
        if self._d_normed:
            return
        d = self._ts['d']
        self._ts['d'] = d / d.norm(2, -1, True)
        self._d_normed = True

    def march_(self, t: Ts, n: float | Ts = None):
        """
        Propagate the rays forward by a distance ``t``. The origin will be updated.

        :param Ts t: Propagation distance. A tensor whose shape must be broadcastable
            with the shape of rays.
        :param n: Refractive index of the medium in where the rays propagate.
            A float or a tensor with shape of rays. Default: 1.
        :type n: float or Tensor
        :return: ``None``
        """
        move = self.d_norm * t.unsqueeze(-1)
        new_o = self._ts['o'] + move
        self._ts['o'] = new_o

        if self._coherent:
            if n is None:
                warnings.warn('Refractive index not specified for coherent rays')
            opl = t if n is None else n * t
            self._ts['opl'] = self._ts['opl'] + opl.unsqueeze(-1)

        valid = self._ts['v']
        new_shape = torch.broadcast_shapes(new_o.shape, valid.shape)
        if new_shape != self.shape:
            self._ts['v'] = torch.broadcast_to(valid, new_shape)
            self._ts['wl'] = torch.broadcast_to(self._ts['wl'], new_shape)
            if self._coherent:
                self._ts['opl'] = torch.broadcast_to(self._ts['opl'], new_shape)

    def to_(self, *args, **kwargs):
        """
        Call :py:meth:`torch.Tensor.to` for all the tensors bound to ``self``
        and update them by the returned tensor.

        This method will not update the dtype of ``self.valid``, which is always
        ``torch.bool``.

        :param args: Positional arguments accepted by :py:meth:`torch.Tensor.to`.
        :param kwargs: Keyword arguments accepted by :py:meth:`torch.Tensor.to`.
        :return: ``None``
        """
        for k in list(self._ts.keys()):
            nt = self._ts[k].to(*args, **kwargs)
            if k == 'v':
                nt = nt.to(torch.bool)
            self._ts[k] = nt

    def update_valid_(self, valid: Ts, action: Literal['discard', 'copy'] = None):
        """
        Update the validity flags with ``valid``. The new validity flag will be
        its logical & with the old.

        :param Tensor valid: A tensor with same shape as that of rays.
        :param str action: The action to take after updating validity flags.
            Call :py:meth:`.discard_` if ``'discard'``, or :py:meth:`.keep_valid_`
            if ``'copy'``, or nothing if ``None``. Default: ``None``.
        :return: ``None``.
        """
        if valid.dtype != torch.bool:
            raise TypeError('Only bool tensors are supported.')
        if valid.shape != self.shape:
            raise ValueError(
                f'The shape of new validity tensor must be same as the old, '
                f'got {valid.shape}'
            )
        self._ts['v'] = torch.logical_and(self._ts['v'], valid)
        if action == 'discard':
            self.discard_()
        elif action == 'copy':
            self.copy_valid_()

    @property
    def shape(self) -> torch.Size:
        """
        The broadcast shape of the ray's origin and direction tensor,
        without the last coordinate dimension.

        :type: torch.Size
        """
        return self._ts['v'].shape

    @property
    def o(self) -> Ts:
        """
        The origin of the rays. A tensor of shape ``(..., 3)``.

        :type: Tensor
        """
        return self._ts['o']

    @o.setter
    def o(self, value: Ts):
        self._check_shape(value)
        self._ts['o'] = value

    @property
    def d(self) -> Ts:
        """
        The direction of the rays. The length or returned direction
        (i.e. 2-norm along the last dimension) may not be normalized to 1.
        Use :py:meth:`norm_d_` to normalize the direction or
        :py:meth:`d_norm` to get a normalized copy of the direction.
        A tensor of shape ``(..., 3)``.

        :type: Tensor
        """
        return self._ts['d']

    @d.setter
    def d(self, value: Ts):
        self._check_shape(value)
        self._ts['d'] = value

    @property
    def d_norm(self) -> Ts:
        """
        The normalized direction of the rays. See :py:meth:`d`
        for the difference between these two.
        A tensor of shape ``(..., 3)``.

        :type: Tensor
        """
        d = self._ts['d']
        return d if self._d_normed else d / d.norm(2, -1, True)

    @property
    def wl(self) -> Ts:
        """
        The wavelengths of rays whose shape is that of rays.

        :type: Tensor
        """
        return self._ts['wl']

    @wl.setter
    def wl(self, value: Ts):
        if value.shape != self.shape:
            raise ValueError(f'Expected shape {self.shape}, got {value.shape}')
        self._ts['wl'] = value

    @property
    def valid(self) -> Ts:
        """
        The validity flag of the rays. A bool tensor whose shape is the shape of rays.

        :type: Tensor
        """
        return self._ts['v']

    @valid.setter
    def valid(self, value: Ts):
        if value.shape != self.shape:
            raise ValueError(f'Expected shape {self.shape}, got {value.shape}')
        if value.dtype != torch.bool:
            raise TypeError(f'bool tensor expected, got dtype {value.dtype}')
        self._ts['v'] = value

    @property
    def opl(self) -> Ts | None:
        """
        The accumulative optical path length of the rays in meters if coherent,
        whose shape is the shape of rays.
        If :py:attr:`.coherent` is ``False``, this property is ``None``.

        :type: Tensor
        """
        return self._ts['opl'] if self._coherent else None

    @opl.setter
    def opl(self, value: Ts):
        if value.shape != self.shape:
            raise ValueError(f'Expected shape {self.shape}, got {value.shape}')
        self._ts['opl'] = value

    @property
    def x(self) -> Ts:
        """
        :math:`x` coordinate of the origin. Its shape is that of ``self.o``
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 0]

    @property
    def y(self) -> Ts:
        """
        :math:`y` coordinate of the origin. Its shape is that of ``self.o``
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 1]

    @property
    def z(self) -> Ts:
        """
        :math:`z` coordinate of the origin. Its shape is that of ``self.o``
        without the last dimension.

        :type: Tensor
        """
        return self._ts['o'][..., 2]

    @property
    def d_x(self) -> Ts:
        """
        The projection of the direction to :math:`x` axis.
        Its shape is that of ``self.d`` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 0]

    @property
    def d_y(self) -> Ts:
        """
        The projection of the direction to :math:`y` axis.
        Its shape is that of ``self.d`` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 1]

    @property
    def d_z(self) -> Ts:
        """
        The projection of the direction to :math:`z` axis.
        Its shape is that of ``self.d`` without the last dimension.

        :type: Tensor
        """
        return self._ts['d'][..., 2]

    def _check_shape(self, vector: Ts):
        if vector.ndim < 1:
            raise ValueError(f'Non-scalar tensor expected, got shape ({vector.shape})')
        if vector.ndim - 1 > len(self.shape):
            raise ValueError(f'At most {len(self.shape) + 1} dimensions expected, '
                             f'got shape ({vector.shape})')
        for d1, d2 in zip(reversed(vector.shape[:-1]), reversed(self.shape)):
            if d1 != d2 and d1 != 1:
                raise ValueError(f'Incompatible shape: {vector.shape}')
