from __future__ import annotations
import pytest
from open_notebook import config

from pathlib import Path
import os
from typing import TYPE_CHECKING

from .utils import run_inside_dir


if TYPE_CHECKING:
    from typing import Generator, TypeVar

    T = TypeVar("T")
    YieldFixture = Generator[T, None, None]


@pytest.fixture(scope="function")
def home_path() -> Path:
    return Path(__file__).parent.absolute() / "data"


@pytest.fixture(scope="function")
def example_path(tmp_path: Path) -> YieldFixture[Path]:
    other_dir = tmp_path / "a" / "b"
    other_dir.mkdir(parents=True)

    # change to example_path
    old_cwd = Path.cwd()
    os.chdir(tmp_path)

    assert Path.cwd().absolute() == tmp_path.absolute()

    yield tmp_path.absolute()
    # Cleanup?
    os.chdir(old_cwd)


@pytest.fixture(scope="function")
def example_path_with_config(example_path: Path) -> YieldFixture[Path]:
    config.create_config(
        host="thing",
        port="8889",
        root="~/Documents",
        dir_prefix="tr",
        file_prefix="no",
        overwrite=True,
        path=example_path,
    )

    yield example_path


@pytest.fixture(scope="function")
def example_path_with_git(example_path: Path) -> YieldFixture[Path]:
    run_inside_dir("git init", example_path)

    yield example_path


@pytest.fixture(scope="function")
def example_path_with_git_config(
    example_path_with_config: Path, example_path_with_git: Path
) -> YieldFixture[Path]:
    assert example_path_with_config == example_path_with_git
    yield example_path_with_config
