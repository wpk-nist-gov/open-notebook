from __future__ import annotations

import pytest

from pathlib import Path

from open_notebook.handler import JupyterUrlHandler


@pytest.fixture
def default_handler() -> JupyterUrlHandler:
    return JupyterUrlHandler(
        root="~/top/level",
        host="localhost",
        port="8888",
        dir_prefix="tree",
        file_prefix="notebooks",
    )


def test_handler_relative_path(default_handler: JupyterUrlHandler) -> None:
    h = default_handler

    assert h._path_relative_to_root(Path("~/top/level/a/b")) == Path("a/b")
    assert h._path_relative_to_root(Path("~/top/level/a/b.txt")) == Path("a/b.txt")

    with pytest.raises(ValueError):
        h._path_relative_to_root(Path("~/top"))


def test_path_to_url(default_handler: JupyterUrlHandler, tmp_path: Path) -> None:
    h = default_handler
    h.root = tmp_path.absolute()

    other_dir = tmp_path / "a" / "b"
    other_dir.mkdir(parents=True)

    with open(other_dir / "tmp.ipynb", "w") as f:  # pyright: ignore
        pass

    assert h.paths_to_urls(other_dir) == ["http://localhost:8888/tree/a/b"]

    assert h.paths_to_urls(other_dir / "tmp.ipynb") == [
        "http://localhost:8888/notebooks/a/b/tmp.ipynb"
    ]

    assert h.paths_to_urls(other_dir / "tmp.ipynb", suffix="?reset") == [
        "http://localhost:8888/notebooks/a/b/tmp.ipynb?reset"
    ]

    with pytest.raises(ValueError):
        h.paths_to_urls(tmp_path / "hello")

    with pytest.raises(ValueError):
        h.paths_to_urls(tmp_path / "hello.ipynb")
