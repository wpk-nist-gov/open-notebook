"""
Top level API (:mod:`open_notebook`)
======================================================
"""

try:
    from ._version import __version__  # type: ignore[unused-ignore,import]
except Exception:  # pragma: no cover
    __version__ = "999"

__author__ = """William P. Krekelberg"""
__email__ = "wpk@nist.gov"


__all__ = [
    "__version__",
]
