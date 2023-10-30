from __future__ import annotations

from open_notebook import config

import pytest
from pathlib import Path

from typing import Any

from .utils import inside_dir


def test_Config_simple() -> None:
    c = config.Config(data={"hello": "there"})

    assert c.get("hello") == "there"
    assert c.get("there", factory=list) == []

    from open_notebook.utils import get_in

    assert get_in(("hello"), {}, factory=list) == []


def base_options(
    host: str = "localhost",
    port: str = "8888",
    root: str = ".",
    dir_prefix: str = "tree",
    file_prefix: str = "notebooks",
) -> dict[str, Any]:
    return dict(
        host=host,
        port=port,
        root=Path(root),
        dir_prefix=dir_prefix,
        file_prefix=file_prefix,
    )


def test_base() -> None:
    c = config.Config([])

    assert base_options() == c.to_options_dict()

    # overriding:

    assert base_options(host="thing") == c.to_options_dict(host="thing")
    assert base_options(port="8889") == c.to_options_dict(port="8889")
    assert base_options(root="~/") == c.to_options_dict(root="~/")
    assert base_options(dir_prefix="tr") == c.to_options_dict(dir_prefix="tr")
    assert base_options(file_prefix="no") == c.to_options_dict(file_prefix="no")


def test_git_root(example_path: Path) -> None:
    assert Path.cwd() == example_path

    assert config.get_git_root_path() is None


def test_git_root_config(example_path_with_config: Path) -> None:
    p = example_path_with_config
    assert Path.cwd() == p

    assert config.get_git_root_path() is None


def test_git_root_git(example_path_with_git: Path) -> None:
    p = example_path_with_git
    assert Path.cwd() == p

    assert config.get_git_root_path() == p


def test_git_root_git_config(example_path_with_git_config: Path) -> None:
    p = example_path_with_git_config
    assert Path.cwd() == p

    assert config.get_git_root_path() == p


def test_find_config(example_path: Path, home_path: Path) -> None:
    out = config.get_config_files(home=home_path)

    assert out["cwd"] is None
    assert out["git"] is None
    assert out["home"] == home_path / config.CONFIG_FILE_NAME

    # empty home:
    out = config.get_config_files(home=Path(__file__).absolute())

    assert out["cwd"] is None
    assert out["git"] is None
    assert out["home"] is None


def test_find_config_with_config(
    example_path_with_config: Path, home_path: Path
) -> None:
    out = config.get_config_files(home=home_path)

    assert out["cwd"] == Path(".").absolute() / config.CONFIG_FILE_NAME
    assert out["git"] is None
    assert out["home"] == home_path / config.CONFIG_FILE_NAME


def test_find_config_with_git(example_path_with_git: Path, home_path: Path) -> None:
    out = config.get_config_files(home=home_path)

    assert out["cwd"] is None
    assert out["git"] is None
    assert out["home"] == home_path / config.CONFIG_FILE_NAME


def test_find_config_with_git_config(
    example_path_with_git_config: Path, home_path: Path
) -> None:
    out = config.get_config_files(home=home_path)

    assert out["cwd"] == example_path_with_git_config / config.CONFIG_FILE_NAME
    assert out["git"] == example_path_with_git_config / config.CONFIG_FILE_NAME
    assert out["home"] == home_path / config.CONFIG_FILE_NAME

    other_dir = example_path_with_git_config / "a" / "b"

    with inside_dir(other_dir):
        out = config.get_config_files(home=home_path)

        assert out["cwd"] is None
        assert out["git"] == example_path_with_git_config / config.CONFIG_FILE_NAME
        assert out["home"] == home_path / config.CONFIG_FILE_NAME

        config.create_config(
            host="localhost",
            port="8888",
            root="~/hello",
            dir_prefix="tr",
            file_prefix="no",
            path=".",
        )

        out = config.get_config_files(home=home_path)

        assert out["cwd"] == other_dir / config.CONFIG_FILE_NAME
        assert out["git"] == example_path_with_git_config / config.CONFIG_FILE_NAME
        assert out["home"] == home_path / config.CONFIG_FILE_NAME

        with pytest.raises(ValueError):
            config.create_config(
                host="localhost",
                port="8888",
                root="~/hello",
                dir_prefix="tr",
                file_prefix="no",
                home=".",
            )


def test_config(example_path: Path, home_path: Path) -> None:
    # no configs
    c = config.Config.from_config_files(home=Path(__file__).parent)

    assert c.to_options_dict() == base_options()

    # single config
    c = config.Config.from_config_files(home=home_path)

    assert c.to_options_dict() == base_options(root="~")


def test_config_with_config(example_path_with_config: Path, home_path: Path) -> None:
    c = config.Config.from_config_files(home=home_path)

    assert c.to_options_dict() == base_options(
        host="thing", port="8889", root="~/Documents", dir_prefix="tr", file_prefix="no"
    )


def test_inheritance() -> None:
    s_home = """
    host = "localhost"
    port = "8889"
    root = "~/other"

    [alt]
    port = "8899"
    """

    c = config.Config.from_strings(s_home)

    assert c.to_options_dict() == base_options(port="8889", root="~/other")
    assert c.to_options_dict(section="alt") == base_options(port="8899", root="~/other")

    s_git = """
    port = "9999"

    [alt2]
    root = "~/thing"
    """

    c = config.Config.from_strings([s_git, s_home])

    assert c.to_options_dict() == base_options(port="9999", root="~/other")
    assert c.to_options_dict(section="alt2") == base_options(
        port="9999", root="~/thing"
    )
