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


def _getattr(obj, attr: str):
    dlg_name: str = getattr(obj, '_delegate_name', '_delegate')
    dlg = getattr(obj, dlg_name, ...)
    if dlg is not ...:
        if callable(dlg):
            dlg = dlg()
        return getattr(dlg, attr)
    raise RuntimeError(f'A delegate object has to be specified to determine {attr}')


class DeviceMixIn:
    _delegate: torch.Tensor
    _delegate_name: str

    @property
    def device(self) -> torch.device:
        """
        Device of this object.

        Some :py:class:`torch.Tensor`s may be associated to an object of the class
        (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
        possessing this property. They are assumed to be on the same device,
        which is the value of this property.

        :type: :py:class:`torch.device`
        """
        return _getattr(self, 'device')


class DtypeMixIn:
    _delegate: torch.Tensor
    _delegate_name: str

    @property
    def dtype(self) -> torch.dtype:
        """
        Data type of this object.

        Some :py:class:`torch.Tensor`s may be associated to an object of the class
        (e.g. buffers and parameters of :py:class:`torch.nn.Module`)
        possessing this property. They are assumed to have same data type,
        which is the value of this property.

        :type: :py:class:`torch.dtype`
        """
        return _getattr(self, 'dtype')


class TensorContainerMixIn(DeviceMixIn, DtypeMixIn):
    pass


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
