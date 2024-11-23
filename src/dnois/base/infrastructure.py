import inspect
import warnings

from .typing import get_overloads

__all__ = [
    'debug',
    'debugging',
    'get_bound_args',
]

_empty = inspect.Parameter.empty
_debug = False


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
