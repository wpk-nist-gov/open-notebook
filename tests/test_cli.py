from __future__ import annotations
from textwrap import dedent

from typing import TYPE_CHECKING
import pytest

from pathlib import Path

from open_notebook import cli
from open_notebook.utils import MISSING

from .utils import run_inside_dir

import shlex

if TYPE_CHECKING:
    from typing import Any
    import argparse
    from open_notebook._typing import MISSING_TYPE


# @pytest.fixture
# def parser() -> ArgumentParser:
#     return cli.get_parser()


def parse_args(arg: str = "") -> argparse.Namespace:
    parser = cli.get_parser()
    return parser.parse_args(shlex.split(arg))


def parse_args_as_dict(arg: str = "") -> dict[str, Any]:
    return vars(parse_args(arg))


def base_cli_options(
    host: str | MISSING_TYPE = MISSING,
    port: str | MISSING_TYPE = MISSING,
    root: str | MISSING_TYPE = MISSING,
    dir_prefix: str | MISSING_TYPE = MISSING,
    file_prefix: str | MISSING_TYPE = MISSING,
    reset: bool = False,
    config: str | None = None,
    create_config: bool = False,
    overwrite: bool = False,
    version: bool = False,
    verbose: int | None = None,
    dry: bool = False,
    paths: list[Path | str] | None = None,
) -> dict[str, Any]:
    if paths is None:
        paths_verified = []
    else:
        paths_verified = list(map(Path, paths))

    return dict(
        host=host,
        port=port,
        root=root,
        dir_prefix=dir_prefix,
        file_prefix=file_prefix,
        reset=reset,
        config=config,
        create_config=create_config,
        overwrite=overwrite,
        version=version,
        verbose=verbose,
        dry=dry,
        paths=paths_verified,
    )


def test_parser() -> None:
    # check missing argument

    with pytest.raises(SystemExit):
        parse_args_as_dict("-d")

    assert parse_args_as_dict("") == base_cli_options()

    assert parse_args_as_dict("--host hello --port 8889") == base_cli_options(
        host="hello", port="8889"
    )

    assert parse_args_as_dict("-p 8889") == base_cli_options(port="8889")

    assert parse_args_as_dict("-r ~/") == base_cli_options(root="~/")
    assert parse_args_as_dict("--root ~/") == base_cli_options(root="~/")

    assert parse_args_as_dict("-c alt") == base_cli_options(config="alt")
    assert parse_args_as_dict("--config alt") == base_cli_options(config="alt")

    assert parse_args_as_dict("--create-config") == base_cli_options(create_config=True)

    assert parse_args_as_dict("-v") == base_cli_options(verbose=1)
    assert parse_args_as_dict("-vv") == base_cli_options(verbose=2)

    assert parse_args_as_dict("--verbose -v") == base_cli_options(verbose=2)

    assert parse_args_as_dict("--dry a/b .") == base_cli_options(
        dry=True, paths=["a/b", "."]
    )


def test_verbosity() -> None:
    import logging

    cli.main([])
    # assert 0 == 0
    assert cli.logger.level == 0

    cli.set_verbosity_level(cli.logger, -1)
    assert cli.logger.level == logging.ERROR

    cli.set_verbosity_level(cli.logger, 0)
    assert cli.logger.level == logging.WARN

    cli.main(["-v"])
    assert cli.logger.level == logging.INFO

    cli.main(["-vv"])
    assert cli.logger.level == logging.DEBUG


def test_version() -> None:
    import open_notebook

    assert cli.get_version_string() == f"open-notebook, {open_notebook.__version__}"


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


def test_get_options(example_path: Path, home_path: Path) -> None:
    cli_options = parse_args("--host hello")

    options = cli.get_options(cli_options, home=home_path)

    assert options == base_options(root="~", host="hello")


def test_get_options_with_config(
    example_path_with_config: Path, home_path: Path
) -> None:
    cli_options = parse_args("--host hello")

    options = cli.get_options(cli_options, home=home_path)

    assert options == base_options(
        host="hello", port="8889", root="~/Documents", dir_prefix="tr", file_prefix="no"
    )


def test_get_options_alt(example_path: Path, home_path: Path) -> None:
    cli_options = parse_args("--config alt")

    s = """\
    [alt]
    host = "test"
    port = "9999"
    """

    with open(".open-notebook.toml", "w") as f:
        f.write(dedent(s))

    # check config files:
    from open_notebook import config

    out = config.get_config_files(home=home_path)

    assert out["cwd"] == example_path / config.CONFIG_FILE_NAME
    assert out["git"] is None
    assert out["home"] == home_path / config.CONFIG_FILE_NAME

    options = cli.get_options(cli_options, home=home_path, section=cli_options.config)

    assert options == base_options(root="~", host="test", port="9999")


def test_create_config0(example_path: Path) -> None:
    options = base_options()
    options["root"] = "."

    cli.create_config(options=options, paths=[Path(".")])

    import tomli

    with open(".open-notebook.toml", "rb") as f:
        data = tomli.load(f)

    assert data == options

    # test that multiple paths creates error

    with pytest.raises(ValueError):
        cli.create_config(options=options, paths=[Path(x) for x in ("a", "a/b")])


def test_create_config1(example_path: Path) -> None:
    options = base_options()
    options["root"] = "."

    cli.create_config(options=options, paths=[], home=".")

    import tomli

    with open(".open-notebook.toml", "rb") as f:
        data = tomli.load(f)

    assert data == options


def test_create_config2(example_path: Path) -> None:
    cli.main(["--create-config", "-r", ".", "."])

    options = base_options()
    # options["root"] = "."

    from open_notebook import config

    c = config.Config.from_paths(".open-notebook.toml")

    assert c.to_options_dict() == options


def test_open(example_path: Path) -> None:
    cli.main(args=["-r", ".", ".", "a/b", "--dry"])
    cli.main(args=["--version"])


def test_run(example_path: Path) -> None:
    from open_notebook import __version__

    for s in [
        "open-notebook --version",
        "nopen --version",
    ]:
        out = run_inside_dir(s)

        assert out.returncode == 0
        assert out.stdout.decode().strip() == f"open-notebook, {__version__}"
