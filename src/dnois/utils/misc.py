import inspect

__all__ = [
    'subclasses',
]


def _subclasses(cls: type) -> set[type]:
    subs = set(cls.__subclasses__())  # use set to avoid duplicates
    for sub in subs.copy():
        subs = subs | _subclasses(sub)
    return subs


def subclasses(cls:type, _filter: bool = True) -> list[type]:
    sub_list = _subclasses(cls)
    if _filter:
        sub_list =  list(filter(lambda c: not inspect.isabstract(c) and not c.__name__.startswith('_'), sub_list))
    sub_list = sorted(sub_list, key=lambda c: c.__name__)
    return sub_list


