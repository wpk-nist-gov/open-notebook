from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    from .utils import _Missing  # pyright: ignore[reportPrivateUsage]

    MISSING_TYPE = Literal[_Missing.MISSING]
