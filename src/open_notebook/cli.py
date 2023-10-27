"""
Program `open-notebook` (:mod:`~open_notebook.cli`)
===================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from open_notebook.utils import MISSING

from .config import DEFAULT_PARAMS

if TYPE_CHECKING:
    from typing import Sequence


import logging

# * Logging
FORMAT = "[%(name)s - %(levelname)s] %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)
logger = logging.getLogger(__name__)


# * Options


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-notebook",
        description="Program to open jupyter notebooks from central notebook server.",
        epilog="""
        You can set options with the configuration files ".open-notebook.toml".  Configuration
        files are found in the current directory, git root (if in a git tracked tree), and the home directory.
        Note that all these files are considered, in order.  That is, you could override a single value in the
        current directory, and the rest would be inherited from, in order, git root and then the home directory.
        """,
    )

    parser.add_argument(
        "--host",
        type=str,
        default=MISSING,
        help="Host name (default='{host}')".format(**DEFAULT_PARAMS),
    )
    parser.add_argument(
        "-p",
        "--port",
        type=str,
        default=MISSING,
        help="Port (default='{port}')".format(**DEFAULT_PARAMS),
    )
    parser.add_argument(
        "-r",
        "--root",
        default=MISSING,
        help="Directory servers was started in. Defaults to current working directory.",
    )
    parser.add_argument(
        "--dir-prefix",
        type=str,
        default=MISSING,
        help="Directory prefix (default='{dir_prefix}')".format(**DEFAULT_PARAMS),
    )
    parser.add_argument(
        "--file-prefix",
        type=str,
        default=MISSING,
        help="File prefix (default='{file_prefix}')".format(**DEFAULT_PARAMS),
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Config style to use.  This is the name of a header in one of the config files.",
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="If passed, create .open-notebook.ini",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass to overwrite `~/.open-notebook.toml` if it exists with `--create-config`",
    )
    parser.add_argument(
        "--version", action="store_true", help="Print out program version"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        help="Set verbosity level.  Can pass multiple times.",
    )
    parser.add_argument("--dry", action="store_true", help="Dry run.")
    parser.add_argument("paths", nargs="*", type=Path, help="file or paths to open")
    return parser


def set_verbosity_level(logger: logging.Logger, verbosity: int | None) -> None:
    if verbosity is None:
        return
    elif verbosity < 0:
        level = logging.ERROR
    elif verbosity == 0:
        level = logging.WARN
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.DEBUG

    logger.setLevel(level)


# * Main program


def get_version_string() -> str:
    from open_notebook import __version__

    return f"open-notebook, {__version__}"


def get_options(
    cli_options: argparse.Namespace,
    section: str | None = None,
    home: str | Path | None = None,
) -> dict[str, Any]:
    from open_notebook import config

    options = config.Config.from_config_files(home=home).to_options_dict(
        section=section, **vars(cli_options)
    )
    logger.debug(f"options: {options}")
    return options


def create_config(
    options: dict[str, Any],
    paths: list[Path],
    overwrite: bool = False,
    home: str | Path | None = None,
) -> None:
    from open_notebook import config

    if (n := len(paths)) == 0:
        p = None
    elif n == 1:
        p = paths[0]
    else:
        raise ValueError("can specify zero or one path for config file")

    config.create_config(**options, overwrite=overwrite, path=p, home=home)


def open_paths(options: dict[str, Any], paths: list[Path], dry: bool = False) -> None:
    from open_notebook import handler

    h = handler.JupyterUrlHandler(**options)

    urls = h.paths_to_urls(paths)

    for url in urls:
        logger.info(f"opening: {url}")
        if not dry:
            handler.open_urls(url)  # pragma: no cover


def main(args: Sequence[str] | None = None, home: str | Path | None = None) -> int:
    """Console script for open_notebook."""

    # get cli options
    parser = get_parser()
    if args is None:
        cli_options = parser.parse_args()  # pragma: no cover
    else:
        cli_options = parser.parse_args(args)

    set_verbosity_level(logger=logger, verbosity=cli_options.verbose)
    logger.debug(f"cli options: {cli_options}")

    if cli_options.version:
        print(get_version_string())

    elif cli_options.create_config:
        create_config(
            options=get_options(cli_options, section=cli_options.config, home=home),
            paths=cli_options.paths,
            overwrite=cli_options.overwrite,
        )

    elif cli_options.paths:
        open_paths(
            options=get_options(cli_options, section=cli_options.config, home=home),
            paths=cli_options.paths,
            dry=cli_options.dry,
        )

    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
