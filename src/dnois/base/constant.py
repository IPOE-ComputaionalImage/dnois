import csv
import importlib.resources

from . import unit as u

__all__ = [
    'fline',
    'fraunhofer_line',
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
                       f'Refer to https://en.wikipedia.org/wiki/Fraunhofer_lines for more information.')

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
