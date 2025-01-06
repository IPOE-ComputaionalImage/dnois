import csv
import importlib.resources

import torch

from . import unit as u, exception
from .typing import Numeric, Ts, overload

__all__ = [
    'fline',
    'fraunhofer_line',
    'reflect',
    'refract',
    'wave_vec',
]

with importlib.resources.open_text(__package__, 'fl.csv') as f:
    _fraunhofer_line_db = [(line[0], line[1], float(line[2]) * 1e-9) for line in csv.reader(f)]


def fraunhofer_line(
    symbol: str = None, element: str = None, alone: bool = True, unit: str = None,
) -> float | dict[str, float] | list[tuple[str, str, float]]:
    """
    Returns information about `Fraunhofer lines <https://en.wikipedia.org/wiki/Fraunhofer_lines>`_.

    The type of return value depends on which arguments are given (when ``alone`` is ``False``):

    - If no arguments are given, returns a list of ``(symbol, element, wavelength)`` tuples
      for all lines.
    - If only ``symbol`` is given, returns a dict whose keys and values are respectively
      elements and wavelengths of lines with given symbol. Note that distinct lines
      may have same symbols.
    - If only ``element`` is given, returns a dict whose keys and values are respectively
      symbols and wavelengths of lines with given element. Note that one element may
      have multiple lines.
    - If both are given, returns a single float indicating corresponding wavelength.

    If ``alone`` is ``True`` and a ``dict`` to be returned contains only one item,
    that wavelength will be returned instead.

    .. testsetup::

        from dnois import fraunhofer_line

    >>> fraunhofer_line('d', 'He')
    5.875618e-07
    >>> fraunhofer_line('d')
    {'He': 5.875618e-07, 'Fe': 4.6681400000000004e-07}
    >>> fraunhofer_line(element='Na')
    {'D_1': 5.89592e-07, 'D_2': 5.889950000000001e-07}
    >>> fraunhofer_line('C')  # alone=True
    6.56281e-07

    .. note::

        The 587.5618nm line has two symbols: :math:`D_3` and :math:`d`.
        Both of them are valid to retrieve this line.

    :param str symbol: Symbol of lines to retrieve.
    :param str element: Element of lines to retrieve.
    :param bool alone: If ``True``, returns wavelength directly instead of a ``dict``
        if just one item matches. Default: ``True``.
    :param str unit: Unit of returned wavelengths. Default: global default unit.
        See :doc:`/content/guide/unit` for more information.
    :return: See description above.
    :rtype: float | dict[str, float] | list[tuple[str, str, float]]
    """
    if unit is None:
        unit = u.get_default_unit()
    if symbol is None:
        if element is None:
            return [(item[0], item[1], u.convert(item[2], 'm', unit)) for item in _fraunhofer_line_db]
        else:
            ret = {item[0]: u.convert(item[2], 'm', unit) for item in _fraunhofer_line_db if item[1] == element}
    elif element is None:
        ret = {item[1]: u.convert(item[2], 'm', unit) for item in _fraunhofer_line_db if item[0] == symbol}
    else:
        for item in _fraunhofer_line_db:
            if item[0] == symbol and item[1] == element:
                return u.convert(item[2], 'm', unit)
        raise KeyError(f'There is no Fraunhofer line with symbol {symbol} and element {element}. '
                       f'Refer to https://en.wikipedia.org/wiki/Fraunhofer_lines or call '
                       f'{fraunhofer_line.__qualname__} without arguments for more information.')

    # ret is a dict
    if len(ret) == 1 and alone:
        return list(ret.values())[0]
    else:
        return ret


def fline(
    symbol: str = None, element: str = None, alone: bool = True, unit: str = None,
) -> float | dict[str, float] | list[tuple[str, str, float]]:
    """Alias for :func:`fraunhofer_line`."""
    return fraunhofer_line(str(symbol), element, alone, unit)


def wave_vec(wl: Numeric) -> Numeric:
    r"""
    Computes magnitude of wavelength vector:

    .. math::
        k=2\pi/\lambda

    :param wl: Wavelength :math:`\lambda`.
    :type wl: Tensor or float
    :return: Magnitude of wavelength vector.
    :rtype: same as ``wl``.
    """
    return torch.pi * 2 / wl


def _as_tensor(x, src: Ts) -> Ts:
    if not torch.is_tensor(x):
        return src.new_tensor(x)
    return x


@overload
def refract(incident: Ts, normal: Ts, mu: Numeric) -> Ts:
    pass


@overload
def refract(incident: Ts, normal: Ts, n1: Numeric, n2: Numeric) -> Ts:
    pass


def refract(incident: Ts, normal: Ts, n1: Numeric, n2: Numeric = None) -> Ts:
    r"""
    Computes direction of refractive ray given that of incident ray,
    normal vector and refractive indices. In descriptions below,
    subscript 1 means incident media and 2 means refractive media.
    All vectors have unit length. :math:`\mathbf{d}` indicates
    directions of rays and :math:`\mathbf{n}` indicates normal vector.

    This function has two overloaded forms:

    -   If relative refractive index :math:`\mu=n_1/n_2` is given, returns

        .. math::

            \mathbf{d}_2=\mu\mathbf{d}_1
            +\sqrt{1-\mu^2[1-(\mathbf{n}\cdot\mathbf{d}_1)^2]}\mathbf{n}
            -\mu(\mathbf{n}\cdot\mathbf{d}_1)\mathbf{n}

    :param Tensor incident: Incident *direction* :math:`\mathbf{d}_1`. A tensor of shape ``(..., 3)``.
    :param Tensor normal: Normal vector :math:`\mathbf{n}`. A tensor of shape ``(..., 3)``.
    :param mu: Relative refractive index :math:`\mu`. A float or a tensor of shape ``(...)``.
    :type mu: float or Tensor
    :return: Refractive *direction* :math:`\mathbf{d}_2`. A tensor of shape ``(..., 3)``.
    :rtype: Tensor

    -   If refractive indices in incident and refractive media :math:`n_1`,
        :math:`n_2` are given, returns

        .. math::

            n_2\mathbf{d}_2=n_1\mathbf{d}_1+
            \left[\sqrt{n_2^2-n_1^2+(\mathbf{n}\cdot n_1\mathbf{d}_1)^2}
            -(\mathbf{n}\cdot n_1\mathbf{d}_1)\right]\mathbf{n}

    :param Tensor incident: Incident *ray vector* :math:`n_1\mathbf{d}_1`. A tensor of shape ``(..., 3)``.
    :param Tensor normal: Normal vector :math:`\mathbf{n}`. A tensor of shape ``(..., 3)``.
    :param n1: Refractive index :math:`n_1`. A float or a tensor of shape ``(...)``.
    :type n1: float or Tensor
    :param n2: Refractive index :math:`n_2`. A float or a tensor of shape ``(...)``.
    :type n2: float or Tensor
    :return: Refractive *ray vector* :math:`n_2\mathbf{d}_2`. A tensor of shape ``(..., 3)``.
    :rtype: Tensor

    .. hint::

        ``n1``, ``n2`` and ``mu`` can be negative.
    """
    ni = torch.sum(normal * incident, -1, True)  # inner product, ... x 1
    if ni.lt(0).any():
        raise exception.PhysicsError('The angle between normal vector and incident ray is not acute.')
    n1 = _as_tensor(n1, ni).unsqueeze(-1)  # ... x 1

    if n2 is None:
        mu = n1  # ... x 1
        nt2 = 1 - mu.square() * (1 - ni.square())  # ... x 1
        refractive = torch.sqrt(nt2) * normal + mu * (incident - ni * normal)
        return refractive
    else:
        ni = ni * n1
        n2 = _as_tensor(n2, ni).unsqueeze(-1)  # ... x 1
        deflection = (torch.sqrt(n2.square() - n1.square() + ni.square()) - ni) * normal
        return incident * n1 + deflection


def reflect(incident: Ts, normal: Ts) -> Ts:
    r"""
    Computes direction of reflective ray given that of incident ray and normal vector:

    .. math::
        \mathbf{d}_2=\mathbf{d}_1-2(\mathbf{n}\cdot\mathbf{d}_1)\mathbf{n}

    :param Tensor incident: Incident direction :math:`\mathbf{d}_1`. A tensor of shape ``(..., 3)``.
    :param Tensor normal: Normal vector :math:`\mathbf{n}`. A tensor of shape ``(..., 3)``.
    :return: Reflective direction :math:`\mathbf{d}_2`. A tensor of shape ``(..., 3)``.
    :rtype: Tensor
    """
    ni = torch.sum(normal * incident, -1, True)  # inner product, ... x 1
    if ni.lt(0).any():
        raise exception.PhysicsError('The angle between normal vector and incident ray is not acute.')
    return incident - 2 * ni * normal
