import inspect
import warnings

import torch

from .typing import Ts, Sequence, get_overloads, overload

__all__ = [
    'broadcastable',
    'get_bound_args',

    'DeviceMixIn',
    'DtypeMixIn',
    'TensorContainerMixIn',
]

_empty = inspect.Parameter.empty


def _check_consistency(attr: str, obj, ts: Ts, error: bool) -> bool:
    v1, v2 = getattr(obj, attr), getattr(ts, attr)
    if v1 != v2:
        if error:
            raise RuntimeError(f'{attr.capitalize()} mismatch: {v1} for an instance of '
                               f'{obj.__class__.__name__} while {v2} for an incoming tensor')
        else:
            return False
    return True


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
    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('device', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(device=self.device)

    @property
    def device(self) -> torch.device:
        """
        Device of this object.

        Some :py:class:`torch.Tensor` s may be associated to an object of the class
        (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
        possessing this property. They are assumed to be on the same device,
        which is the value of this property.

        :type: :py:class:`torch.device`
        """
        return self._delegate().device


class DtypeMixIn(TensorAsDelegate):
    def _check_consistency(self, ts: Ts, error: bool = True) -> bool:
        return _check_consistency('dtype', self, ts, error)

    def _cast(self, ts: Ts) -> Ts:
        return ts.to(dtype=self.dtype)

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of this object.

        Some :py:class:`torch.Tensor` s may be associated to an object of the class
        (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
        possessing this property. They are assumed to have same data type,
        which is the value of this property.

        :type: :py:class:`torch.dtype`
        """
        return self._delegate().dtype


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


def get_bound_args(func, *args, **kwargs) -> inspect.BoundArguments:
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


def broadcastable(*tensors_or_shapes) -> bool:
    if torch.is_tensor(tensors_or_shapes[0]):
        shapes = [t.shape for t in tensors_or_shapes]
    else:
        shapes = tensors_or_shapes
    try:
        torch.broadcast_shapes(*shapes)
    except RuntimeError:
        return False
    else:
        return True
