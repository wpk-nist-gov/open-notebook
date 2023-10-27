from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

R = TypeVar("R")

if TYPE_CHECKING:
    from typing_extensions import Literal

    from .utils import _Missing

    MISSING_TYPE = Literal[_Missing.MISSING]
