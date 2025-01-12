from functools import partial

import torch

from .. import utils
from ..base.serialize import AsJsonMixIn
from ..base.typing import Ts, Numeric, Any, Callable, overload, cast

__all__ = ['Transform']

TransFn = Callable[[Ts], Ts]


def _ts_or_float(x: Numeric) -> Ts | float:
    if isinstance(x, float) or torch.is_tensor(x):
        return x
    else:
        return torch.tensor(x)


class Transform(AsJsonMixIn):
    """
    Base class for parameter transformations (see :class:`~dnois.torch.ParamTransformModule`
    and :doc:`transformed parameters </content/guide/transform>`). This class also serves
    as a namespace containing functions to create commonly used transformations.
    All these function return a transformation object.

    The constructor of this class has two overloaded forms:

    - Accepts an inverse transformation after ``fn`` or as ``inverse`` argument;
    - Accepts any arguments combination (``*args`` and ``**kwargs``) which will be
      passed to ``fn`` along with latent value to calculate nominal value.
      No inverse transformation is specified in this case.

    In function descriptions below, :math:`x` indicates latent value and :math:`y`
    indicates nominal value.
    """

    @overload
    def __init__(self, fn: TransFn, inverse: TransFn = None):
        pass

    @overload
    def __init__(self, fn: TransFn, *args, **kwargs):
        pass

    def __init__(self, fn: TransFn, *args, **kwargs):
        if self.__class__ is Transform and fn is None:
            raise TypeError("Transform function fn must not be None")
        inverse = self._extract_inverse(*args, **kwargs)
        self._transform = partial(fn, *args, **kwargs) if inverse is None and fn is not None else fn
        self._inverse = inverse

    def transform(self, x: Ts) -> Ts:
        return self._transform(x)

    def inverse(self, y: Ts) -> Ts:
        return self._inverse(y)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        raise NotImplementedError(f'General transformations cannot be converted to a dict')

    @property
    def invertible(self):
        return self.inverse is not None

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Transform:
            d.pop('type')
            return cls(**d)

        ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == ty:
                return cast(type[Transform], sub).from_dict(d)
        types = [sub.__name__ for sub in subs]
        raise RuntimeError(f'Unknown transform type: {ty}. Available: {types}')

    @staticmethod
    def scale(s: Numeric) -> 'Transform':
        """
        Multiply the latent value by a scalar factor:

        .. math::
            y=sx,x=y/s

        :param s: Scale factor :math:`s`.
        :type s: float or Tensor
        """
        return Scale(s)

    @staticmethod
    def range(min_: Numeric, max_: Numeric) -> 'Transform':
        r"""
        Limit the range of parameter to :math:`(a,b)` using sigmoid function:

        .. math::
            y=(b-a)\sigma(x)+a,x=\sigma^{-1}(\frac{y-a}{b-a})

        where :math:`\sigma(t)=\frac{1}{1+\e^{-t}}` and :math:`\sigma^{-1}(t)=\ln\frac{t}{1-t}`.

        :param min_: Lower bound :math:`a`.
        :type min_: float or Tensor
        :param max_: Upper bound :math:`b`.
        :type max_: float or Tensor
        """
        return Range(min_, max_)

    @staticmethod
    def positive() -> 'Transform':
        r"""
        Force parameter to be positive using exponential function:

        .. math::
            y=\e^x,x=\ln y
        """
        return Gt()

    @staticmethod
    def negative() -> 'Transform':
        r"""
        Force parameter to be negative using exponential function:

        .. math::
            y=-\e^x,x=\ln -y
        """
        return Lt()

    @staticmethod
    def gt(limit: Numeric) -> 'Transform':
        r"""
        Force parameter to be greater than :math:`a` using exponential function:

        .. math::
            y=a+\e^x,x=\ln(y-a)

        :param limit: Lower bound :math:`a`.
        :type limit: float or Tensor
        """
        return Gt(limit)

    @staticmethod
    def lt(limit: Numeric) -> 'Transform':
        r"""
        Force parameter to be less than :math:`b` using exponential function:

        .. math::
            y=b-\e^x,x=\ln(b-y)

        :param limit: Upper bound :math:`b`.
        :type limit: float or Tensor
        """
        return Lt(limit)

    @staticmethod
    def composite(*transforms: 'Transform') -> 'Transform':
        r"""
        Chain some transformations into a single transformation.
        Resulted transformation is to apply all the transformations sequentially
        and resulted inverse transformation is to apply all the inversions in a reversed order.

        :param Transform transforms: One or more transformations to apply.
        """
        return Composite(*transforms)

    @staticmethod
    def _extract_inverse(*args, **kwargs):
        if 'inverse' in kwargs:
            if len(args) + len(kwargs) > 1:
                raise ValueError(f'No more arguments is acceptable when inverse is given')
            return kwargs['inverse']
        elif len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return args[0]
        else:
            return None


# These classes are keep not public to make namespace clean
class Scale(Transform):
    def __init__(self, s: Numeric):
        super().__init__(cast(Callable, None))
        self._s = _ts_or_float(s)

    def transform(self, x: Ts) -> Ts:
        return x * self._s

    def inverse(self, y: Ts) -> Ts:
        return y / self._s

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            's': self._attr2dictitem('_s', keep_tensor)
        }


class Range(Transform):
    def __init__(self, min_: Numeric, max_: Numeric):
        super().__init__(cast(Callable, None))
        self._min = _ts_or_float(min_)
        self._range = _ts_or_float(max_) - self._min

    def transform(self, x: Ts) -> Ts:
        return self._min + self._range * x.sigmoid()

    def inverse(self, y: Ts) -> Ts:
        return torch.logit((y - self._min) / self._range)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        _min = self._attr2dictitem('_min', keep_tensor)
        return {
            'type': self.__class__.__name__,
            'min_': _min,
            'max_': self._attr2dictitem('_range', keep_tensor) + _min,
        }


class Gt(Transform):
    def __init__(self, limit: Numeric = None):
        super().__init__(cast(Callable, None))
        self._limit = _ts_or_float(limit)

    def transform(self, x: Ts) -> Ts:
        return x.exp() if self._limit is None else self._limit + x.exp()

    def inverse(self, y: Ts) -> Ts:
        return y.log() if self._limit is None else torch.log(y - self._limit)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'limit': self._attr2dictitem('_limit', keep_tensor),
        }


class Lt(Transform):
    def __init__(self, limit: Numeric = None):
        super().__init__(cast(Callable, None))
        self._limit = _ts_or_float(limit)

    def transform(self, x: Ts) -> Ts:
        return -x.exp() if self._limit is None else self._limit - x.exp()

    def inverse(self, y: Ts) -> Ts:
        return y.neg().log() if self._limit is None else torch.log(self._limit - y)

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'limit': self._attr2dictitem('_limit', keep_tensor),
        }


class Composite(Transform):
    def __init__(self, *transforms: Transform):
        super().__init__(cast(Callable, None))
        self._transforms = transforms

    def transform(self, x: Ts) -> Ts:
        for t in self._transforms:
            x = t.transform(x)
        return x

    def inverse(self, y: Ts) -> Ts:
        for t in reversed(self._transforms):
            y = t.inverse(y)
        return y

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'transforms': [t.to_dict(keep_tensor) for t in self._transforms]
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(*[Transform.from_dict(item) for item in d['transforms']])
