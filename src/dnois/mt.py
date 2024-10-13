"""
This module provides a series of classes representing various optical materials
with several different dispersion formula, such as Cauchy formula, Schott formula
and Sellmeier formula, etc.

All the material classes are derived from :py:class:`Material` and share its
constructor arguments. They have also a method :py:meth:`~Material.n`
to compute the refractive index for given wavelength. Each class implements
this method by its own dispersion formula.

This module maintains a material library to add, delete or retrieve materials.
See :ref:`accessing_materials`.
"""
# TODO: declare source: Zemax manual

import abc

import torch

from .base import convert
from .base.typing import Numeric, Union, cast

__all__ = [
    'vacuum',

    'get',
    'is_available',
    'list_all',
    'refractive_index',
    'register',
    'remove',

    'Cauchy',
    'Conrady',
    'Constant',
    'Herzberger',
    'Material',
    'Schott',
    'Sellmeier1',
    'Sellmeier2',
    'Sellmeier3',
    'Sellmeier4',
    'Sellmeier5',
]

_PRINT_PRECISION = 3


def _format_flist(flist: list[float], precision: int) -> str:
    return f'[{", ".join(f"{c:.{precision}f}" for c in flist)}]'


class Material(metaclass=abc.ABCMeta):
    """
    Class representing an optical material type.

    :param str name: Name of the material.
    :param float min_wl: Minimum applicable wavelength in ``default_unit``. Default: 0.
    :param float max_wl: Maximum applicable wavelength in ``default_unit``. Default: infinity.
    :param str default_unit: Default unit for wavelength.
    """
    __slots__ = ('name', 'min_wl', 'max_wl', 'default_unit')

    def __init__(self, name: str, min_wl: float = None, max_wl: float = None, default_unit: str = 'm'):
        #: Name of the material.
        self.name = name
        #: Minimum wavelength valid for the material.
        self.min_wl = min_wl if min_wl is not None else 0.
        #: Maximum wavelength valid for the material.
        self.max_wl = max_wl if max_wl is not None else float('inf')
        #: Default unit.
        self.default_unit = default_unit

    def __repr__(self):
        return f'{self.__class__.__name__}({self._repr(_PRINT_PRECISION)})'

    @abc.abstractmethod
    def n(self, wavelength: Numeric, unit: str = None) -> Numeric:
        """
        Computes refractive index.

        :param wavelength: Value of wavelength.
        :type: float or Tensor
        :param str unit: The unit of given wavelength. Default: ``self.default_unit``.
        :return: Refractive index.
        :rtype: float or Tensor
        """
        pass

    @abc.abstractmethod
    def _repr(self, precision: int) -> str:
        return (f'name={self.name}, '
                f'domain=({self.min_wl:.{precision}f}, {self.max_wl:.{precision}f}), '
                f'in {self.default_unit}')

    def _validate(self, wl: Numeric, unit: str = None) -> Numeric:
        # convert wl in unit to wl in default unit if unit is given,
        # or wl is assumed to be with default unit
        if unit is not None and unit != self.default_unit:
            wl = convert(wl, unit, self.default_unit)

        m1, m2 = (wl.min().item(), wl.max().item()) if torch.is_tensor(wl) else (wl, wl)
        if m1 < self.min_wl * (1 - 1e-6) or m2 > self.max_wl * (1 + 1e-6):
            raise ValueError(
                f'Unsupported wavelength for material \'{self.name}\': '
                f'{wl}(unit: {unit or self.default_unit})'
            )
        else:
            return wl


class Constant(Material):
    """
    Material with constant optical properties.

    :param float n: Constant refractive index.

    See :class:`Material` for descriptions for other parameters.
    """

    def __init__(
        self,
        name: str,
        n: float,
        min_wl: float = None,
        max_wl: float = None,
        default_unit: str = 'm'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.refractive_index: float = n  #: Refractive index.

    def _repr(self, precision: int) -> str:
        return f'n={self.refractive_index:.{precision}f}'

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        n = self.refractive_index
        return torch.full_like(wl, n) if torch.is_tensor(wl) else n


class Cauchy(Material):
    r"""
    Material for which Cauchy formula:

    .. math::
        n=A+\frac{B}{\lambda^2}+\frac{C}{\lambda^4}

    :param float a: :math:`A` in Cauchy formula.
    :param float b: :math:`B` in Cauchy formula.
    :param float c: :math:`C` in Cauchy formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a', 'b', 'c')

    def __init__(
        self, name: str, a: float, b: float, c: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a = a  #: :math:`A` in Cauchy formula.
        self.b = b  #: :math:`B` in Cauchy formula.
        self.c = c  #: :math:`C` in Cauchy formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        iw2 = 1 / wl ** 2
        n = (self.c * iw2 + self.b) * iw2 + self.a
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'A={self.a:.{precision}f}, '
                f'B={self.b:.{precision}f}, '
                f'C={self.c:.{precision}f}')


class Schott(Material):
    r"""
    Materials described by Schott formula:

    .. math::
        n^2=a_0+a_1\lambda^2+a_2\lambda^{-2}+a_3\lambda^{-4}+a_4\lambda^{-6}+a_5\lambda^{-8}

    :param list[float] coefficients: The six coefficients in Schott formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('coefficients',)

    def __init__(
        self, name: str, coefficients: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Schott formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Schott formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        iw2 = 1 / wl ** 2
        n2 = self.coefficients.pop(-1)
        for c in reversed(self.coefficients):
            n2 = n2 * iw2 + c
        n = n2 ** 0.5
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'coefficients={_format_flist(self.coefficients, precision)}')


class _Sellmeier(Material):
    __slots__ = ('ks', 'ls')
    _n_terms: int

    def __init__(
        self, name: str, ks: list[float], ls: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

        if len(ks) != self._n_terms or len(ls) != self._n_terms:
            raise ValueError(f'Numbers of K and L coefficients should be equal to {self._n_terms}.')
        self.ks = ks  #: The coefficients :math:`K_i` s in Sellmeier{num} formula.
        self.ls = ls  #: The coefficients :math:`L_i` s in Sellmeier{num} formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        w2 = wl ** 2
        n2 = 1 + sum([kc * w2 / (w2 - lc) for kc, lc in zip(self.ks, self.ls)])
        n = n2 ** 0.5
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'K={_format_flist(self.ks, precision)}, '
                f'L={_format_flist(self.ls, precision)}')


def _make_sellmeier(num: int, n_terms: int) -> type[_Sellmeier]:
    cls = type(f'Sellmeier{num}', (_Sellmeier,), {'_n_terms': n_terms})
    cls.__doc__ = fr"""
    Materials described by Sellmeier{num} formula:

    .. math::
        n^2-1=\sum_{{i=1}}^{n_terms}\frac{{K_i\lambda^2}}{{\lambda^2-L_i}}
        
    :param list[float] ks: The coefficients :math:`K_i,i=1,\cdots,{n_terms}` in Sellmeier{num} formula.
    :param list[float] ls: The coefficients :math:`L_i,i=1,\cdots,{n_terms}` in Sellmeier{num} formula.
    
    See :class:`Material` for descriptions for other parameters.
    """
    return cast(type[_Sellmeier], cls)


Sellmeier1 = _make_sellmeier(1, 3)
Sellmeier3 = _make_sellmeier(3, 4)
Sellmeier5 = _make_sellmeier(5, 5)


class Sellmeier2(Material):
    r"""
    Materials described by Sellmeier2 formula:

    .. math::
        n^2-1=A+\frac{B_1\lambda^2}{\lambda^2-\lambda_1^2}+\frac{B_2}{\lambda^2-\lambda_2^2}

    :param float a: :math:`A` in Sellmeier2 formula.
    :param float b1: :math:`B_1` in Sellmeier2 formula.
    :param float b2: :math:`B_2` in Sellmeier2 formula.
    :param float wl1: :math:`\lambda_1` in Sellmeier2 formula.
    :param float wl2: :math:`\lambda_2` in Sellmeier2 formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a_pp', 'b1', 'b2', 'swl1', 'swl2')

    def __init__(
        self, name: str, a: float, b1: float, b2: float, wl1: float, wl2: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a_pp = a + 1  #: :math:`A+1` in Sellmeier2 formula.
        self.b1 = b1  #: :math:`B_1` in Sellmeier2 formula.
        self.b2 = b2  #: :math:`B_2` in Sellmeier2 formula.
        self.swl1 = wl1 * wl1  #: :math:`\lambda_1^2` in Sellmeier2 formula.
        self.swl2 = wl2 * wl2  #: :math:`\lambda_2^2` in Sellmeier2 formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        w2 = wl ** 2
        n2 = self.a_pp + self.b1 * w2 / (w2 - self.swl1) + self.b2 / (w2 - self.swl2)
        n = n2 ** 0.5
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'A={self.a_pp - 1:.{precision}f}, '
                f'B1={self.b1:.{precision}f}, B2={self.b2:.{precision}f}, '
                f'Wl1={self.swl1 ** 0.5:.{precision}f}, Wl2={self.swl2 ** 0.5:.{precision}f}')


class Sellmeier4(Material):
    r"""
    Materials described by Sellmeier4 formula:

    .. math::
        n^2=A+\frac{B\lambda^2}{\lambda^2-C}+\frac{D\lambda^2}{\lambda^2-E}

    :param float a: :math:`A` in Sellmeier4 formula.
    :param float b: :math:`B` in Sellmeier4 formula.
    :param float c: :math:`C` in Sellmeier4 formula.
    :param float d: :math:`D` in Sellmeier4 formula.
    :param float e: :math:`E` in Sellmeier4 formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('a', 'b', 'c', 'd', 'e')

    def __init__(
        self, name: str, a: float, b: float, c: float, d: float, e: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.a = a  #: :math:`A` in Sellmeier4 formula.
        self.b = b  #: :math:`B` in Sellmeier4 formula.
        self.c = c  #: :math:`C` in Sellmeier4 formula.
        self.d = d  #: :math:`D` in Sellmeier4 formula.
        self.e = e  #: :math:`E` in Sellmeier4 formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        w2 = wl ** 2
        n2 = self.a + self.b * w2 / (w2 - self.c) + self.d * w2 / (w2 - self.e)
        n = n2 ** 0.5
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'A={self.a:.{precision}f}, '
                f'B={self.b:.{precision}f}, '
                f'C={self.c:.{precision}f}, '
                f'D={self.d:.{precision}f}, '
                f'E={self.e:.{precision}f}')


class Herzberger(Material):
    r"""
    Materials described by Herzberger formula:

    .. math::
        n=A+BL+CL^2+D\lambda^2+E\lambda^4+F\lambda^6,\\
        L=\frac{1}{\lambda^2-0.028}

    :param list[float] coefficients: The six coefficients in Herzberger formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('coefficients',)

    def __init__(
        self, name: str, coefficients: list[float],
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)

        if len(coefficients) != 6:
            raise ValueError(f'Number of coefficients in Herzberger formula must be 6.')
        self.coefficients = coefficients  #: The six coefficients in Herzberger formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        w2 = wl ** 2
        m = 1 / (w2 - 0.028)
        _1, _2, _3, _4, _5, _6 = self.coefficients
        n = _1 + m * (_2 + m * _3) + w2 * (_4 + w2 * (_5 + w2 * _6))
        return n

    def _repr(self, precision: int) -> str:
        return f'{super()._repr(precision)}, coefficients={_format_flist(self.coefficients, precision)}'


class Conrady(Material):
    r"""
    Materials described by Conrady formula:

    .. math::
        n=n_0+\frac{A}{\lambda}+\frac{B}{\lambda^{3.5}}

    :param float n0: :math:`n_0` in Conrady formula.
    :param float a: :math:`A` in Conrady formula.
    :param float b: :math:`B` in Conrady formula.

    See :class:`Material` for descriptions for other parameters.
    """
    __slots__ = ('n0', 'a', 'b')

    def __init__(
        self, name: str, n0: float, a: float, b: float,
        min_wl: float = None, max_wl: float = None, default_unit: str = 'um'
    ):
        super().__init__(name, min_wl, max_wl, default_unit)
        self.n0 = n0  #: :math:`n_0` in Conrady formula.
        self.a = a  #: :math:`a` in Conrady formula.
        self.b = b  #: :math:`b` in Conrady formula.

    def n(self, wl: Numeric, unit: str = None) -> Numeric:
        wl = self._validate(wl, unit)
        n = self.n0 + self.a / wl + self.b / wl ** 3.5
        return n

    def _repr(self, precision: int) -> str:
        return (f'{super()._repr(precision)}, '
                f'n0={self.n0:.{precision}f}, '
                f'A={self.a:.{precision}f}, '
                f'B={self.b:.{precision}f}')


vacuum: Constant = Constant('vacuum', 1.)  #: Vacuum.
_lib = {
    'vacuum': vacuum,
}


def get(name: str, default_none: bool = False) -> Union[Material, None]:
    """
    Get material by name from material library.

    :param str name: Name of the material.
    :param bool default_none: If true, return ``None`` when the material does not exist.
        Otherwise, an ``ValueError`` is raised.
    :return: Specified material.
    :rtype: Material
    """
    m = _lib.get(name, None)
    if m is None:
        if default_none:
            return None
        raise KeyError(f'Unknown material: {name}')
    return m


def register(name: str, material: Material):
    """
    Add a new class of material into material library.

    :param str name: Name of the material.
    :param Material material: The material instance.
    """
    if name in _lib:
        raise KeyError(f'Material {name} already exists.')
    _lib[name] = material


def refractive_index(wavelength: Numeric, material: str, unit='m') -> Numeric:
    """
    Compute refractive index for given wavelength and material.

    :param wavelength: Specified wavelength.
    :type: float or Tensor
    :param str material: Specified material.
    :param str unit: Unit of wavelength.
    :return: Refractive index.
    :rtype: float or Tensor
    """
    m = get(material)
    n = m.n(wavelength, unit)
    return n


def is_available(name: str) -> bool:
    """
    Check if given material is available in material library.

    :param str name: Name of the material.
    :return: If the material is available.
    :rtype: bool
    """
    return name in _lib


def list_all() -> list[str]:
    """
    List all available materials in material library.

    :return: List of the names of available materials.
    :rtype: list[str]
    """
    return list(_lib.keys())


def remove(name: str, ignore_if_absent: bool = False):
    """
    Remove a material from material library.

    :param str name: Name of the material.
    :param bool ignore_if_absent: If true, ignore material if it does not exist.
        A :py:exc:`KeyError` will be raised otherwise.
    :return: None
    """
    if name in _lib:
        del _lib[name]
    elif not ignore_if_absent:
        raise KeyError(f'Unknown material: {name}')
