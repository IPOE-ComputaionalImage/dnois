from dnois.base.typing import Any

from .lg import Surface

__all__ = [
    'build_surface',
]


def build_surface(surface_config: dict[str, Any]) -> Surface:
    raise NotImplementedError()
