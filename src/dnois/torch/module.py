from functools import partial, wraps

import torch
from torch import nn

from .param_transform import Transform
from ..base.serialize import AsJsonMixIn
from ..base.typing import Callable, Ts, cast

__all__ = [
    'DeviceMixIn',
    'DtypeMixIn',
    'EnhancedModule',
    'ParamTransformModule',
    'TensorContainerMixIn',
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
    has same effect. Transformations are specified in the form of a :class:`Transform` instance.

    .. note::
        **Implementation Detail** The latent value of each transformed parameter is
        a :class:`torch.nn.Parameter` attribute with name like ``_latent_<param>``.
    """
    _param_transforms: dict[str, Transform]

    def __getattr__(self, name: str):
        param_transforms = cast(dict, self.__dict__.get('_param_transforms', _unset))
        if param_transforms is _unset:
            return super().__getattr__(name)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            return param_transforms[name].transform(super().__getattr__(latent_name))
        else:
            return super().__getattr__(name)

    def __setattr__(self, name, value):
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return super().__setattr__(name, value)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            transform = param_transforms[name]
            if not transform.invertible:
                raise RuntimeError(f'No inverse transformation specified for transformed parameter {name} '
                                   f'so it cannot be assigned. Assign to its latent "{latent_name}" '
                                   f'if you intend to modify its value directly.')
            else:
                super().__setattr__(latent_name, transform.inverse(value))
        else:
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return super().__delattr__(name)

        latent_name = self._latent_name(name)
        if name in param_transforms:
            super().__delattr__(latent_name)
            del param_transforms[name]
        else:
            return super().__delattr__(name)

    def register_parameter(
        self, name: str, param: nn.Parameter | None, transform: Transform = None
    ) -> None:
        """
        Similar to :meth:`torch.nn.Module.register_parameter`, but allows you to register a transformed
        parameter as long as ``transform`` is given.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        but ``transform`` is given, it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Nominal :class:`torch.nn.Parameter` instance to be registered.
        :param Transform transform: Transformation object.
        """
        if transform is None:
            return super().register_parameter(name, param)
        if not transform.invertible:
            raise ValueError(f'Inverse transformation cannot be None when registering parameter {name} '
                             f'with transformation given. Call register_latent_parameter to register '
                             f'a transformed parameter without inverse transformation.')

        if param is not None:
            param.data = transform.inverse(param.data)
        return self.register_latent_parameter(name, param, transform)

    def register_latent_parameter(
        self, name: str, param: nn.Parameter | None, transform: Transform
    ):
        """
        Similar to :meth:`.register_parameter`, but takes as input the latent value
        rather than nominal value. In this way, the inverse transformation need not be
        provided since the initial latent value is known.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Parameter param: Latent :class:`torch.nn.Parameter` instance to be registered.
        :param Transform transform: Transformation object.
        """
        param_obj = getattr(self, name, _unset)
        if param_obj is not _unset:
            if not isinstance(param_obj, torch.nn.Parameter):
                raise AttributeError(
                    f'Attribute {name} of class {self.__class__.__name__} is not a torch.nn.Parameter'
                )
            delattr(self, name)

        lt_name = self._latent_name(name)
        self.register_parameter(lt_name, param)  # method of super class will be called virtually

        transforms = self._get_transforms_dict()
        transforms[name] = transform

    def set_transform(self, name: str, transform: Transform):
        """
        Set transformation for parameter ``name``.

        If ``name`` corresponds to a vanilla parameter (i.e. not transformed parameter)
        it will be converted to a transformed one.

        :param str name: Name of the parameter.
        :param Transform transform: Transformation object.
        """
        transforms = self._get_transforms_dict()
        if name in transforms:
            lt_name = self._latent_name(name)
            return self.register_latent_parameter(name, getattr(self, lt_name), transform)
        return self.register_parameter(name, getattr(self, name), transform)

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
                    value = named_params.pop(k)
                    named_params[name] = param_transforms[name].transform(value)
        return named_params

    @property
    def transformed_parameters(self) -> dict[str, tuple[torch.nn.Parameter, Transform]]:
        """
        A ``dict`` whose keys are names of all transformed parameters of this module.
        The value corresponding to each key is a tuple containing:

        - The latent value, a :class:`torch.nn.Parameter` instance;
        - Corresponding transformation object.

        :type: dict[str, tuple[Parameter, Transform]]
        """
        param_transforms = getattr(self, '_param_transforms', _unset)
        if param_transforms is _unset:
            return {}
        return {
            name: (getattr(self, self._latent_name(name)), tr)
            for name, tr in param_transforms.items()
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


class EnhancedModule(ParamTransformModule, AsJsonMixIn):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_to_dict = cls.__dict__.get('to_dict', None)
        original_from_dict = cls.__dict__.get('from_dict', None)

        if original_to_dict is not None:
            original_to_dict = cast(Callable, original_to_dict)

            def _wrapped_to_dict(self, keep_tensor: bool = True):
                self: EnhancedModule
                d = original_to_dict(self, keep_tensor)
                tp = self.transformed_parameters
                if len(tp) != 0:
                    d['transformed_parameters'] = {k: (
                        self._attr2dictitem(self._latent_name(k), keep_tensor),
                        v[1].to_dict(keep_tensor)
                    ) for k, v in tp.items()}
                return d

            cls.to_dict = wraps(original_to_dict)(_wrapped_to_dict)

        if original_from_dict is not None:
            if isinstance(original_from_dict, classmethod):
                original_from_dict = original_from_dict.__wrapped__

            def _wrapped_from_dict(clz, d: dict):
                clz: type[EnhancedModule]
                tp = d.pop('transformed_parameters', {})
                obj = original_from_dict(clz, d)
                for k, (param, transform) in tp.items():
                    obj.register_latent_parameter(k, param, Transform.from_dict(transform))
                return obj

            cls.from_dict = classmethod(wraps(original_from_dict)(_wrapped_from_dict))


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
