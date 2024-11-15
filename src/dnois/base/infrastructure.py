import inspect
import warnings

import torch

from .typing import Ts, Sequence, get_overloads, overload

__all__ = [
    'broadcastable',
    'debug',
    'debugging',
    'get_bound_args',

    'DeviceMixIn',
    'DtypeMixIn',
    'TensorContainerMixIn',
]

_empty = inspect.Parameter.empty
_debug = False


def _check_consistency(attr: str, obj, ts: Ts, error: bool) -> bool:
    v1, v2 = getattr(obj, attr), getattr(ts, attr)
    if v1 != v2:
        if error:
            raise RuntimeError(f'{attr.capitalize()} mismatch: {v1} for an instance of '
                               f'{obj.__class__.__name__} while {v2} for an incoming tensor')
        else:
            return False
    return True


def debug(on: bool = True):
    """
    Switch :doc:`debugging mode </content/guide/debug>` on or off.

    :param bool on: Enable debugging mode if ``True``, disable it otherwise. Default: ``True``.
    :return: ``None``
    """
    global _debug
    _debug = bool(on)


def debugging() -> bool:
    """
    Returns whether debugging is enabled or not.

    :return: Whether debugging is enabled or not.
    :rtype: bool
    """
    return _debug


class TensorAsDelegate:
    def new_tensor(self, data, **kwargs) -> Ts:
        return self._delegate().new_tensor(data, **kwargs)

    def new_full(self, size, fill_value, **kwargs) -> Ts:
        return self._delegate().new_full(size, fill_value, **kwargs)

    def new_empty(self, size, **kwargs) -> Ts:
        return self._delegate().new_empty(size, **kwargs)

    def new_ones(self, size, **kwargs) -> Ts:
        return self._delegate().new_ones(size, **kwargs)

    def new_zeros(self, size, **kwargs) -> Ts:
        return self._delegate().new_zeros(size, **kwargs)

    def _delegate(self) -> Ts:
        raise TypeError(f'No delegate attribute specified for class {self.__class__.__name__}')


class DeviceMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to be on the same device,
    which is the value of :attr:`device`.
    """

    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('device', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(device=self.device)

    @property
    def device(self) -> torch.device:
        """
        Device of this object.

        :type: :py:class:`torch.device`
        """
        dlg = self._delegate()
        # torch.get_default_device() is not available for old versions
        return torch.tensor(0.).device if dlg is None else dlg.device


class DtypeMixIn(TensorAsDelegate):
    """
    Some :py:class:`torch.Tensor` s may be associated to objects of the class
    (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
    derived from this class. They are assumed to have same data type,
    which is the value of :attr:`dtype`.
    """

    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('dtype', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of this object.

        :type: :py:class:`torch.dtype`
        """
        dlg = self._delegate()
        return torch.get_default_dtype() if dlg is None else dlg.dtype


class TensorContainerMixIn(DeviceMixIn, DtypeMixIn):
    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return (_check_consistency('device', self, ts, error) and
                _check_consistency('dtype', self, ts, error))

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(device=self.device, dtype=self.dtype)


def _match_annotation(ba: inspect.BoundArguments, params) -> bool:
    for name, value in ba.arguments.items():
        param: inspect.Parameter = params[name]
        annt = param.annotation
        if annt is not _empty and not isinstance(value, annt):
            return False
    return True


def get_bound_args(func, *args, **kwargs) -> inspect.BoundArguments:  # check: no use currently
    ols = get_overloads(func)
    if not ols:
        warnings.warn(f'Trying to {get_bound_args.__name__} on a function without overloads')
        return inspect.signature(func).bind(*args, **kwargs)

    self = getattr(func, '__self__', None)
    if self is not None:
        args = (self,) + args
    for ol in ols:
        sig = inspect.signature(ol)
        try:
            ba = sig.bind(*args, **kwargs)
        except TypeError:
            continue
        else:
            if _match_annotation(ba, sig.parameters):
                return ba
    raise TypeError(f'Cannot find a valid overload of {func.__name__} to bind arguments to')


@overload
def broadcastable(*tensors: Ts) -> bool:
    pass


@overload
def broadcastable(*shapes: Sequence[int]) -> bool:
    pass


def broadcastable(*tensors_or_shapes: Sequence[int] | Ts) -> bool:
    r"""
    Check whether some tensors or some tensor shapes are broadcastable.

    :param tensors_or_shapes: Some tensors or some tensor shapes.
    :return: Whether they are broadcastable.
    :rtype: bool
    """
    is_tensor = all(torch.is_tensor(x) for x in tensors_or_shapes)
    is_shape = all(isinstance(x, Sequence) for x in tensors_or_shapes)
    if not is_tensor and not is_shape:
        raise TypeError(f'Arguments of {broadcastable.__name__} must be all tensors or all shapes, '
                        f'but got types {[type(x) for x in tensors_or_shapes]}')
    if is_tensor:
        shapes = [t.shape for t in tensors_or_shapes]
    else:
        shapes = tensors_or_shapes
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError:
        return False
    else:
        return True
