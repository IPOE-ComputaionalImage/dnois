from functools import partial

import torch
from torch import nn

from ..base.typing import Callable, Ts, Numeric, overload, cast

__all__ = [
    'DeviceMixIn',
    'DtypeMixIn',
    'ParamTransformModule',
    'TensorContainerMixIn',
    'Transforms',
    'WrapperModule',
]


class WrapperModule(nn.Module):
    """
    A class to wrap a function as a :py:class:`torch.nn.Module`.

    .. doctest::
        :skipif: True

        >>> s = WrapperModule(torch.sum, dim=(-2, -1))
        >>> x = torch.rand(4)
        >>> s(x)  # equivalent to torch.sum(x, dim=(-2, -1))

    :param Callable func: The function to be wrapped.
    :param args: Positional arguments to be passed to ``func`` when this module is called.
    :param kwargs: Keyword arguments to be passed to ``func`` when this module is called.
    """

    def __init__(self, func: Callable, *args, **kwargs):
        super().__init__()
        self._impl = partial(func, *args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Call the wrapped function ``func``.

        :param args: Additional positional arguments to be passed to ``func``.
        :param kwargs: Additional keyword arguments to be passed to ``func``.
        :return: The returned value of the wrapped function.
        :rtype: Any
        """
        return self._impl(*args, **kwargs)


_unset = object()
Transform = Callable[[Ts], Ts]


class ParamTransformModule(nn.Module):
    """
    A subclass of :class:`torch.nn.Module` that can create and manage
    :doc:`transformed parameters </content/guide/transform>`.

    Transformed parameter can be registered given either its nominal value
    (by calling overridden :meth:`.register_parameter`) or latent value (by calling
    :meth:`.register_latent_parameter`). Note that in first case an inverse transformation
    must be given to compute the latent value. If a parameter with same name has been
    registered already, it will be automatically converted to a transformed parameter.
    Specifying a transformation for existent parameter by calling :meth:`.set_transform`
    has same effect.

    All transformations and their inversion should be a callable object
    that takes a single tensor as input and returns a single tensor.

    .. note::
        **Implementation detail** The latent value of each transformed parameter is
        a :class:`torch.nn.Parameter` attribute with name like ``_latent_<param>``.
    """
    _param_transforms: dict[str, tuple[Transform | None, Transform | None]]

    def __getattr__(self, name: str):
        param_transforms = cast(dict, self.__dict__.get('_param_transforms', _unset))
        if param_transforms is _unset:
            return super().__getattr__(name)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            return param_transforms[name][0](super().__getattr__(latent_name))
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return super().__setattr__(name, value)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            transform = param_transforms[name]
            if transform[1] is None:
                raise RuntimeError(f'No inverse transformation specified for transformed parameter {name} '
                                   f'so it cannot be assigned. Assign to its latent "{latent_name}" '
                                   f'if you intend to modify its value directly.')
            else:
                super().__setattr__(latent_name, transform[1](value))
        return super().__setattr__(name, value)

    def __delattr__(self, name):
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return super().__delattr__(name)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            super().__delattr__(latent_name)
            del param_transforms[name]
        return super().__delattr__(name)

    def register_parameter(
        self, name: str, param: nn.Parameter | None, transform: Transform = None, inverse: Transform = None,
    ) -> None:
        """
        Similar to :meth:`torch.nn.Module.register_parameter`, but allows you to register a transformed
        parameter as long as ``transform`` and ``inverse`` are given.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Nominal :class:`torch.nn.Parameter` instance to be registered.
        :param transform: Transformation to calculate nominal value from latent value.
        :param inverse: Inverse transformation to calculate latent value from nominal value.
        """
        if transform is None:
            if inverse is None:
                return super().register_parameter(name, param)
            else:
                raise ValueError(f'transform cannot be None for transformed parameter {name}')
        elif inverse is None:
            raise ValueError(f'Inverse transformation cannot be None when registering parameter {name} '
                             f'with transformation given. Call register_latent_parameter to register '
                             f'a transformed parameter without inverse transformation.')

        if param is not None:
            param.data = inverse(param.data)
        return self.register_latent_parameter(name, param, transform, inverse=inverse)

    @overload
    def register_latent_parameter(
        self, name: str, param: nn.Parameter | None, fn: Transform, inverse: Transform = None
    ):
        pass

    @overload
    def register_latent_parameter(
        self, name: str, param: nn.Parameter | None, fn: Callable[[Ts, ...], Ts] | str, *args, **kwargs
    ):
        pass

    def register_latent_parameter(
        self, name: str, param: nn.Parameter | None, fn: Callable[[Ts, ...], Ts] | str, *args, **kwargs
    ):
        """
        Similar to :meth:`.register_parameter`, but takes as input the latent value
        rather than nominal value. In this way, the inverse transformation need not be
        provided since the initial latent value is known.

        This method has two overloaded forms:

        - Accepts an inverse transformation after ``fn`` or as ``inverse`` argument;
        - Accepts any arguments combination (``*args`` and ``**kwargs``) which will be
          passed to ``fn`` along with latent value to calculate nominal value.
          No inverse transformation is specified in this case.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Latent :class:`torch.nn.Parameter` instance to be registered.
        :param fn: Transformation to calculate nominal value from latent value.
        """
        inverse = self._extract_inverse(*args, **kwargs)
        if inverse is not None:
            del kwargs['inverse']

        param_obj = getattr(self, name, _unset)
        if param_obj is not _unset:
            if not isinstance(param_obj, torch.nn.Parameter):
                raise AttributeError(
                    f'Attribute {name} of class {self.__class__.__name__} is not a torch.nn.Parameter'
                )
            delattr(self, name)

        lt_name = self._latent_name(name)
        self.register_parameter(lt_name, param)  # method of super class will be called finally

        transforms = self._get_transforms_dict()
        transforms[name] = (partial(fn, *args, **kwargs), inverse)

    @overload
    def set_transform(self, name: str, fn: Transform, inverse: Transform = None):
        pass

    @overload
    def set_transform(self, name: str, fn: Transform, *args, **kwargs):
        pass

    def set_transform(self, name: str, fn: Transform, *args, **kwargs):
        """
        Set transformation, and optionally, its inversion for parameter ``name``.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        See :meth:`.register_latent_parameter` for overloaded forms.

        :param str name: Name of the parameter.
        :param fn: Transformation to calculate nominal value from latent value.
        """
        transforms = self._get_transforms_dict()
        if name in transforms:
            lt_name = self._latent_name(name)
            return self.register_latent_parameter(name, getattr(self, lt_name), fn, *args, **kwargs)

        inverse = self._extract_inverse(*args, **kwargs)
        return self.register_parameter(name, getattr(self, name), fn, inverse)

    @property
    def nominal_values(self) -> dict[str, Ts]:
        """
        A ``dict`` whose keys are names of all parameters of this module
        and values are their values. The values are nominal ones for transformed parameters.

        :type: dict[str, Tensor]
        """
        named_params = list(self.named_parameters(recurse=False))
        named_params = {item[0]: item[1] for item in named_params}
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return named_params

        for k in list(named_params.keys()):
            if k.startswith('_latent_'):
                name = self._nominal_name(k)
                if name in param_transforms:
                    del named_params[k]
                    named_params[name] = param_transforms[name][0](named_params[k])
        return named_params

    @property
    def transformed_parameters(self) -> dict[str, tuple[torch.nn.Parameter, Transform, Transform | None]]:
        """
        A ``dict`` whose keys are names of all transformed parameters of this module.
        The value corresponding to each key is a tuple containing:

        - The latent value, a :class:`torch.nn.Parameter` instance;
        - Corresponding transformation;
        - Corresponding inverse transformation, ``None`` if it is not specified.

        :type: dict[str, tuple[Parameter, Callable, Callable or None]]
        """
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return {}
        return {
            name: (getattr(self, self._latent_name(name)), fn, inv)
            for name, (fn, inv) in param_transforms
        }

    def _get_transforms_dict(self):
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            param_transforms = {}
            self._param_transforms = param_transforms
        return param_transforms

    @staticmethod
    def _latent_name(nominal_name: str) -> str:
        return f'_latent_{nominal_name}'

    @staticmethod
    def _nominal_name(latent_name: str) -> str:
        return latent_name[8:]

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


class Transforms:
    """
    A namespace containing functions to create commonly used transformations
    and their inversion for :class:`ParamTransformModule`. Each function returns
    a 2-tuple of functions, representing a transformation and its inversion.

    In function descriptions below, :math:`x` indicates latent value and :math:`y`
    indicates nominal value.
    """

    @staticmethod
    def scale(s: Numeric) -> tuple[Transform, Transform]:
        """
        Multiply the latent value by a scalar factor:

        .. math::
            y=sx,x=y/s

        :param s: Scale factor :math:`s`.
        :type s: float or Tensor
        """
        return lambda x: x * s, lambda y: y / s

    @staticmethod
    def range(min_: Numeric, max_: Numeric) -> tuple[Transform, Transform]:
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
        range_ = max_ - min_
        return lambda x: min_ + range_ * x.sigmoid(), lambda y: torch.logit((y - min_) / range_)

    @staticmethod
    def positive() -> tuple[Transform, Transform]:
        r"""
        Force parameter to be positive using exponential function:

        .. math::
            y=\e^x,x=\ln y
        """
        return lambda x: x.exp(), lambda y: y.log()

    @staticmethod
    def negative() -> tuple[Transform, Transform]:
        r"""
        Force parameter to be negative using exponential function:

        .. math::
            y=-\e^x,x=\ln -y
        """
        return lambda x: -x.exp(), lambda y: y.neg().log()

    @staticmethod
    def gt(limit: Numeric) -> tuple[Transform, Transform]:
        r"""
        Force parameter to be greater than :math:`a` using exponential function:

        .. math::
            y=a+\e^x,x=\ln(y-a)

        :param limit: Lower bound :math:`a`.
        :type limit: float or Tensor
        """
        return lambda x: limit + x.exp(), lambda y: torch.log(y - limit)

    @staticmethod
    def lt(limit: Numeric) -> tuple[Transform, Transform]:
        r"""
        Force parameter to be less than :math:`b` using exponential function:

        .. math::
            y=b-\e^x,x=\ln(b-y)

        :param limit: Upper bound :math:`b`.
        :type limit: float or Tensor
        """
        return lambda x: limit - x.exp(), lambda y: torch.log(limit - y)


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
