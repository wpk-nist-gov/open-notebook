"""
Url creation/opening (:mod:`~open_notebook.handler`)
====================================================
"""

from __future__ import annotations

from os.path import relpath
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Iterable


class JupyterUrlHandler:
    """Class to create and open urls for use with jupyter notebook server."""

    def __init__(
        self, root: str | Path, host: str, port: str, dir_prefix: str, file_prefix: str
    ) -> None:
        if isinstance(root, str):
            root = Path(root)

        self.root = root.expanduser().absolute()
        self.host = host
        self.port = port
        self.dir_prefix = dir_prefix
        self.file_prefix = file_prefix

    def _path_relative_to_root(self, path: Path) -> Path:
        path = path.expanduser().absolute()

        if (self.root not in path.parents) and (self.root != path):
            raise ValueError(f"path {path} is not a subpath of root {self.root}.")

        return Path(relpath(path.expanduser().absolute(), start=self.root))

    def _path_to_url(self, path: Path, modifier: str, suffix: str | None = None) -> str:
        path = self._path_relative_to_root(path)
        out = f"http://{self.host}:{self.port}/{modifier}/{path}"
        if suffix:
            out += suffix
        return out

    def paths_to_urls(
        self, paths: Path | str | Iterable[str | Path], suffix: str | None = None
    ) -> list[str]:
        if isinstance(paths, (Path, str)):
            paths = [paths]

        urls: list[str] = []
        for path in map(Path, paths):
            if path.is_dir():
                url = self._path_to_url(path, modifier=self.dir_prefix, suffix=None)
            elif path.is_file():
                url = self._path_to_url(path, modifier=self.file_prefix, suffix=suffix)
            else:
                raise ValueError(f"path {path} is not a file or directory")
            urls.append(url)

        return urls

    def open_paths(
        self,
        paths: Path | str | Iterable[str | Path],
        suffix: str | None = None,
        new: int = 0,
        autoraise: bool = True,
    ) -> None:
        open_urls(  # pragma: no cover
            urls=self.paths_to_urls(paths=paths, suffix=suffix),
            new=new,
            autoraise=autoraise,
        )


def open_urls(
    urls: str | Iterable[str], new: int = 0, autoraise: bool = True
) -> None:  # pragma: no cover
    import webbrowser

    if isinstance(urls, str):
        urls = [urls]

    for url in urls:
        webbrowser.open(url, new=new, autoraise=autoraise)
