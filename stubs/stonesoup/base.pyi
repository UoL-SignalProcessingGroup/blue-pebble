"""Partial type stub for ``stonesoup.base``, read by pyrefly via ``search-path``.

Stone Soup ships no type information, so a checker sees ``x: int = Property(...)`` as assigning
a ``Property`` to an ``int``. Typing ``Property`` as returning ``Any`` lets each declaration keep
its annotation without suppressing assignment checks elsewhere.

Only the names bluepebble imports are declared. ``Base`` is not marked with
``dataclass_transform`` because Stone Soup allows a property without a default to follow one
with a default, which dataclass semantics reject.
"""

from abc import ABCMeta
from typing import Any

def Property(
    cls: Any = None,
    *,
    default: Any = ...,
    default_factory: Any = ...,
    doc: str | None = None,
    readonly: bool = False,
    allow_none_with_factory: bool = False,
) -> Any: ...

class BaseMeta(ABCMeta): ...

class Base(metaclass=BaseMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
