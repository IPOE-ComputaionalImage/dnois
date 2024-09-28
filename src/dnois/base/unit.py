from .typing import Numeric

__all__ = [
    'units',

    'convert',
    'scale',
]

units = {
    'm': 1.,
    'dm': 1e-1,
    'cm': 1e-2,
    'mm': 1e-3,
    'um': 1e-6,
    'nm': 1e-9,
    'pm': 1e-12,
    'A': 1e-10
}
_global_unit_length = 'm'


def scale(name: str) -> float:
    v = units.get(name, None)
    if v is None:
        raise ValueError(f'Unknown unit: {name}')
    return v


def convert(value: Numeric, from_: str, to: str) -> Numeric:
    ratio = scale(from_) / scale(to)
    value = value * ratio
    return value
