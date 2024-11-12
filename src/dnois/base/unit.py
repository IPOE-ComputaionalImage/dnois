from .typing import Numeric

__all__ = [
    'convert',
    'get_default_unit',
    'scale',
    'set_default_unit',
    'units',
]

_length_units = {
    'm': 1.,
    'dm': 1e-1,
    'cm': 1e-2,
    'mm': 1e-3,
    'um': 1e-6,
    'nm': 1e-9,
    'pm': 1e-12,
    'A': 1e-10
}
_global_length_unit = 'm'


def units() -> list[str]:
    """
    Returns a list of all available length units.

    :return: Available length units.
    :rtype: list[str]
    """
    return list(_length_units.keys())


def get_default_unit() -> str:
    """
    Get global default length unit.

    :return: Global default length unit.
    :rtype: str
    """
    return _global_length_unit


def set_default_unit(unit: str):
    """
    Set global default length unit.

    :param str unit: Specified global default length unit.
    """
    if unit not in _length_units:
        raise ValueError(f'Unknown length unit: {unit}')


def scale(unit: str) -> float:
    v = _length_units.get(unit, None)
    if v is None:
        raise ValueError(f'Unknown unit: {unit}')
    return v


def convert(value: Numeric, from_: str, to: str) -> Numeric:  # trailing underline due to "from" is a keyword
    """
    Convert a quantity with given unit ``from_`` to that with unit ``to``.

    :param value: The quantity to be converted.
    :type value: float | Tensor
    :param str from_: Original unit of ``value``.
    :param str to: Target unit of ``value``.
    :return: Converted quantity.
    :rtype: float | Tensor
    """
    ratio = scale(from_) / scale(to)
    value = value * ratio
    return value
