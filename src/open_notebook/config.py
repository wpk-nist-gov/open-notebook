"""
Configuration file routines (:mod:`~open_notebook.config`)
==========================================================
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from .utils import MISSING, get_in

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable

    from typing_extensions import Self

    from ._typing import MISSING_TYPE


# * Parameters
CONFIG_FILE_NAME = ".open-notebook.toml"

DEFAULT_PARAMS = {
    "host": "localhost",
    "port": "8888",
    "root": ".",
    "dir_prefix": "tree",
    "file_prefix": "notebooks",
}


def get_git_root_path(cwd: str | Path | None = None) -> Path | None:
    if cwd:
        cwd = Path(cwd).expanduser().absolute()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, cwd=cwd
    )
    out = result.stdout.decode().rstrip()
    if out:
        return Path(out).absolute()
    else:
        return None


def get_config_files(
    cwd: str | Path = ".",
    home: str | Path | None = None,
    config_name: str = CONFIG_FILE_NAME,
) -> dict[str, Path | None]:
    """
    Find the config file

    Order for search is "current directory",  "git root", "user home".

    Parameters
    ----------
    cwd : str or Path, default="."
        Current directory.
    home : str or Path, optional
        Defaults to ``pathlib.Path.home``.  For testing purposes.
    config_name : str, default=".open-notebook.ini"
    Name of config file
    """

    out: dict[str, Path | None] = {}

    cwd = Path(cwd).expanduser().absolute()

    def _has_config(d: str | Path | None) -> Path | None:
        if d is None:
            return None

        d = Path(d)

        p = d / config_name
        if p.exists():
            return p.absolute()
        else:
            return None

    # check current directory
    out["cwd"] = _has_config(cwd)

    # check git root
    out["git"] = _has_config(get_git_root_path(cwd))

    # check home
    out["home"] = _has_config(home or Path.home())

    return out


# * Config
class Config:
    """
    Configuration handler

    Accepts multiple mappings.  Will parse them left to right for matches
    """

    def __init__(
        self,
        data: Mapping[str, Any] | Iterable[Mapping[str, Any]],
        default_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.data: Iterable[Mapping[str, Any]]
        if isinstance(data, Mapping):
            self.data = [data]  # pyright: ignore
        else:
            self.data = data

        if default_params is None:
            default_params = DEFAULT_PARAMS
        self.default_params = default_params

    # def append(self, data: Mapping[str, Any], inplace: bool = False) -> Self:
    #     if inplace:
    #         self.data.append(data)
    #         return self
    #     else:
    #         self.data + [data]
    #         return type(self)(data)

    def get(
        self,
        *keys: str,
        default: Any = MISSING,
        factory: Callable[[], Any] | None = None,
    ) -> Any:
        """Get value from config(s)"""
        for data in self.data:
            out = get_in(keys, data, default=MISSING)
            if out is not MISSING:
                return out

        # else use default
        if factory is not None:
            return factory()  # pragma: no cover
        else:
            return default

    def get_option(
        self,
        key: str,
        passed: Any = MISSING,
        section: str | None = None,
        default: Any = MISSING,
        factory: Callable[[], Any] | None = None,
    ) -> Any:
        """
        Get option value from either passed in option, or from config file (left to right).

        Order of checking is:

        * `passed`
        * `data[section][key]` for `data` in `self.data`.
        * `data[key]` for `data` in `self.data`.
        * factory
        * default

        Parameters
        ----------
        key : str, optional
            key to look for.
        passed : default=MISSING
            Passed in value.  If not MISSING, this will be the returned value.
        section : str, optional
            The table section to check.  Fall back to top level of any dict.
        factory : callable, optional
            If no value found, return `factory()`.
        default :
            Fallback value to return

        """

        if passed is not MISSING:
            return passed

        elif section is not None:
            out = self.get(section, key)
            if out is not MISSING:
                return out

        # if default is MISSING, fallback to DEFAULT_PARAMS
        if default is MISSING:
            default = self.default_params[key]

        # Fall back to top level or default_params
        return self.get(key, default=default, factory=factory)

    ## * Options
    def host(
        self,
        section: str | None = None,
        passed: Any = MISSING,
        default: str | MISSING_TYPE = MISSING,
    ) -> str:
        return self.get_option(  # type: ignore
            section=section, key="host", passed=passed, default=default
        )

    def port(
        self,
        section: str | None = None,
        passed: Any = MISSING,
        default: str | MISSING_TYPE = MISSING,
    ) -> str:
        return self.get_option(  # type: ignore
            section=section, key="port", passed=passed, default=default
        )

    def root(
        self,
        section: str | None = None,
        passed: Any = MISSING,
        default: str | MISSING_TYPE = MISSING,
    ) -> Path:
        return (
            Path(
                self.get_option(
                    section=section, key="root", passed=passed, default=default
                )
            )
            # .expanduser()
            # .absolute()
        )

    def dir_prefix(
        self,
        section: str | None = None,
        passed: Any = MISSING,
        default: str | MISSING_TYPE = MISSING,
    ) -> str:
        return self.get_option(  # type: ignore
            section=section, key="dir_prefix", passed=passed, default=default
        )

    def file_prefix(
        self,
        section: str | None = None,
        passed: Any = MISSING,
        default: str | MISSING_TYPE = MISSING,
    ) -> str:
        return self.get_option(  # type: ignore
            section=section, key="file_prefix", passed=passed, default=default
        )

    def to_options_dict(self, section: str | None = None, **kws: Any) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k in ["host", "port", "root", "dir_prefix", "file_prefix"]:
            out[k] = getattr(self, k)(section=section, passed=kws.get(k, MISSING))
        return out

    ## * Factory
    # specialty stuff
    @classmethod
    def from_paths(
        cls,
        paths: str | Path | Iterable[str | Path],
        default_params: Mapping[str, Any] | None = None,
    ) -> Self:
        import tomli

        if isinstance(paths, (str, Path)):
            paths = [paths]  # pragma: no cover

        data: list[dict[str, Any]] = []
        for path in map(Path, paths):
            with path.open("rb") as f:
                data.append(tomli.load(f))

        return cls(data, default_params=default_params)

    @classmethod
    def from_strings(
        cls,
        strings: str | Iterable[str],
        default_params: Mapping[str, Any] | None = None,
    ) -> Self:
        import tomli

        if isinstance(strings, str):
            strings = [strings]

        data: list[dict[str, Any]] = []
        for string in strings:
            data.append(tomli.loads(string))
        return cls(data, default_params=default_params)

    @classmethod
    def from_config_files(
        cls,
        name: str = CONFIG_FILE_NAME,
        cwd: str | Path = ".",
        home: str | Path | None = None,
        default_params: Mapping[str, Any] | None = None,
    ) -> Self:
        config_path_dict = get_config_files(cwd=cwd, home=home, config_name=name)

        paths: list[str | Path] = []
        for k in ["cwd", "git", "home"]:
            v = config_path_dict[k]
            if v is not None:
                paths.append(v)
        return cls.from_paths(paths, default_params=default_params)


## * Create config
def create_config(
    host: str,
    port: str,
    root: str | Path,
    dir_prefix: str,
    file_prefix: str,
    path: str | Path | None = None,
    overwrite: bool = False,
    home: str | Path | None = None,
) -> None:
    """Create a configuration file."""
    if path is None:
        if home is None:
            home = Path.home()  # pragma: no cover
        else:
            home = Path(home).expanduser().absolute()
        path = home / CONFIG_FILE_NAME
    else:
        path = Path(path)

        assert (
            path.is_dir()
        ), f"Can only specify a directory to place the config file {CONFIG_FILE_NAME} into"

        path = path / CONFIG_FILE_NAME

    if path.exists() and not overwrite:
        raise ValueError(
            f"file {path} exists.  Either remove this file or specify overwrite"
        )

    from textwrap import dedent

    out = f"""\
    # This is the config file for "open-notebook"
    #
    # These options are top level
    host = "{host}"
    port = "{port}"
    root = "{root}"
    dir_prefix = "{dir_prefix}"
    file_prefix = "{file_prefix}"

    # # You can specify other "configurations" using a section.
    # # For example, if you have the following, then calling
    # #
    # # $ open-notebook -c alt ...
    # #
    # # Will use port `8889` and root `~/Documents`, with other options inherited from base config.
    # #
    # [alt]
    # port = "8889"
    # root = "~/Documents"
    """

    with path.open("w") as f:
        f.write(dedent(out))
