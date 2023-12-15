"""Utilities to work with nox"""
from __future__ import annotations

import os
import shlex
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, cast

# fmt: off
from nox.sessions import SessionRunner

# Override SessionRunner._create_venv
_create_venv_super = SessionRunner._create_venv

def override_sessionrunner_create_venv(self) -> None:
    """Override SessionRunner._create_venv"""

    if callable(self.func.venv_backend):
        # if passed a callable backend, always use just that
        # i.e., don't override
        logger.info("Using custom callable venv_backend")

        self.venv = self.func.venv_backend(
            location=self.envdir,
            interpreter=self.func.python,
            reuse_existing=self.func.reuse_venv or self.global_config.reuse_existing_virtualenvs,
            venv_params=self.func.venv_params,
            runner=self,
        )
        return
    else:
        logger.info("Using nox venv_backend")
        return _create_venv_super(self)

SessionRunner._create_venv = override_sessionrunner_create_venv
# fmt: on

import nox.command
from nox.logger import logger
from nox.virtualenv import CondaEnv

if TYPE_CHECKING:
    import sys
    from typing import Any, Callable, Iterable, Literal, TextIO, Union

    from nox import Session

    PathLike = Union[str, Path]

    if sys.version_info < (3, 11):
        from typing_extensions import Self
    else:
        from typing import Self


class VenvBackend:
    """Create a callable venv_backend."""

    def __init__(
        self,
        parse_posargs: Callable[..., dict[str, Any]],
        envname: str | None = None,
        requirements: str | Path | None = None,
        backend: Literal[
            "conda", "mamba", "micromamba", "virtualenv", "venv"
        ] = "conda",
        lock_fallback: bool | None = None,
    ) -> None:
        self.envname = envname
        self.backend = backend
        self.parse_posargs = parse_posargs
        self.lock_fallback = lock_fallback or self.is_conda()

        self._requirements = requirements

    def set_opts(self, *posargs: str) -> None:
        self.opts: dict[str, Any] = self.parse_posargs(*posargs)

    @property
    def lock(self) -> bool:
        return cast(bool, self.opts.get("lock", False))

    @property
    def update(self) -> bool:
        return cast(bool, self.opts.get("update", False))

    def is_conda(self):
        return self.backend in {"conda", "mamba", "micromamba"}

    def set_requirements(self, python_version: str | None) -> None:
        if self._requirements is not None:
            r = Path(self._requirements)
        elif self.envname is None:
            r = None
        else:
            if self.is_conda():
                assert (
                    python_version is not None
                ), "Must pass python version for conda env"

            r = Path(
                infer_requirement_path(
                    name=self.envname,
                    python_version=python_version,
                    ext=".yaml" if self.is_conda() else ".txt",
                    lock=self.lock,
                    lock_fallback=self.lock_fallback,
                )
            )

        self.requirements = r

    def set_tmp_path(self, location: str) -> None:
        self.tmp_path = Path(location) / "tmp"

    def set_params_from_runner(self, runner: SessionRunner) -> None:
        self.set_opts(*runner.posargs)
        self.set_requirements(python_version=runner.func.python)  # type: ignore
        self.set_tmp_path(location=runner.envdir)

    @property
    def hash_path(self) -> Path:
        if self.requirements is None:
            raise ValueError("trying to use hash_path with no requirements")
        return self.tmp_path / (self.requirements.name + ".hash.json")

    @property
    def config_path(self) -> Path:
        return self.tmp_path / "env.json"

    def create_conda_env(self, runner: SessionRunner, reuse_existing: bool) -> CondaEnv:
        venv = CondaEnv(
            location=runner.envdir,
            interpreter=runner.func.python,  # type: ignore
            reuse_existing=reuse_existing,
            venv_params=runner.func.venv_params,
            conda_cmd=self.backend.replace("micro", ""),
        )
        if not self.requirements:
            venv.create()
            return venv

        create = venv._clean_location()

        with check_for_change_manager(
            self.requirements,
            hash_path=self.hash_path,
            force_write=create or self.update,
        ) as changed:
            if create:
                cmds = ["create"]
            elif changed or self.update:
                # recreate
                self.config_path.unlink(missing_ok=True)
                cmds = ["update", "--prune"]
            else:
                # reuse
                cmds = ["reuse"]
                venv._reused = True

            # create environment
            cmd = cmds[0]
            logger.info(f"{cmd.capitalize()} conda environment in {venv.location_name}")
            if cmd != "reuse":
                cmds = (
                    [self.backend, "env"]
                    # + ([] if self.backend == "micromamba" else ["env"])
                    + cmds
                    + [
                        "--yes",
                        "--prefix",
                        venv.location,
                        "--file",
                        str(self.requirements),
                    ]
                )
                if venv_params := runner.func.venv_params:
                    cmds.extend(venv_params)

                logger.info(" ".join(cmds))
                nox.command.run(cmds, silent=False, log=nox.options.verbose or False)
        return venv

    # def create_virtualenv(self, runner: SessionRunner, reuse_existing: bool) -> VirtualEnv:
    #     venv =

    def __call__(
        self,
        location: str,
        interpreter: str | None,
        reuse_existing: bool,
        venv_params: Any,
        runner: SessionRunner,
    ) -> CondaEnv:
        self.set_params_from_runner(runner)

        if self.is_conda():
            return self.create_conda_env(runner=runner, reuse_existing=reuse_existing)
        else:
            raise ValueError


def conda_venv_backend(
    location: str,
    interpreter: str | None,
    reuse_existing: bool,
    venv_params: Any,
    runner: SessionRunner,
) -> CondaEnv:
    """
    Custom venv_backend to create conda environment from `environment.yaml` file.

    venv_params should be a dict with following keys:

    * 'envname' : "name" of environment
    * 'parse_posargs' :  parse_posargs(*posargs) -> dict[str, Any].
    * 'conda_backend' : {"mamba", "conda"}
    * 'lock_fallback' : bool (whether to lock_fallback to non-locked)

    Sets `env_file = requirements/py{py}-{envname}`.

    The environment is cached to `location/tmp/{env_file}.hash.json`
    """
    if not interpreter:
        raise ValueError("must supply interpreter for this backend")

    # get envname, parse_posargs, lock_fallback, and venv_params
    assert isinstance(venv_params, dict)
    assert (envname := venv_params.get("envname")) is not None
    assert (parse_posargs := venv_params.get("parse_posargs")) and callable(
        parse_posargs
    )
    assert isinstance(lock_fallback := venv_params.get("lock_fallback", True), bool)
    assert (conda_backend := venv_params.get("conda_backend", "conda")) in {
        "mamba",
        "conda",
    }

    # options
    opts = cast("dict[str, Any]", parse_posargs(*runner.posargs))

    venv = CondaEnv(
        location=location,
        interpreter=interpreter,
        reuse_existing=reuse_existing,
        venv_params=venv_params.get("venv_params"),
    )

    env_file = infer_requirement_path(
        name=envname,
        ext=".yaml",
        python_version=interpreter,
        lock=opts.get("lock", False),
        lock_fallback=lock_fallback,
    )

    # hash of env_file
    tmp_path = Path(location) / "tmp"
    hash_path = tmp_path / (Path(env_file).name + ".hash.json")
    changed, hashes, _ = check_hash_path_for_change(env_file, hash_path=hash_path)

    # Custom creating (based on CondaEnv.create)
    if not venv._clean_location():
        if changed or opts.get("update", False):
            # remove "env.json" if it exists:
            if (p := tmp_path / "env.json").is_file():
                p.unlink()
            cmd, extras = "update", "--prune"
        else:
            cmd, extras = "reuse", ""
            venv._reused = True
    else:
        cmd, extras = "create", ""

    # create environment
    logger.info(f"{cmd.capitalize()} conda environment in {venv.location_name}")
    if cmd != "reuse":
        cmd = f"{conda_backend} env {cmd} --prefix {venv.location} --file {env_file} {extras}"
        logger.info(cmd)
        nox.command.run(shlex.split(cmd), silent=True, log=nox.options.verbose or False)

    tmp_path.mkdir(parents=True, exist_ok=True)
    write_hashes(hash_path, hashes)

    return venv


# * Top level installation functions ---------------------------------------------------
def py_prefix(python_version: Any) -> str:
    if isinstance(python_version, str):
        return "py" + python_version.replace(".", "")
    else:
        raise ValueError(f"passed non-string value {python_version}")


def _verify_path(
    path: PathLike,
) -> str:
    if not Path(path).is_file():
        raise ValueError(f"Path {path} is not a file")
    return str(path)


def _verify_paths(
    paths: PathLike | Iterable[PathLike] | None,
) -> list[str]:
    if paths is None:
        return []
    elif isinstance(paths, (str, Path)):
        paths = [paths]

    return [_verify_path(p) for p in paths]


def infer_requirement_path(
    name: str | None,
    ext: str | None = None,
    python_version: str | None = None,
    lock: bool = False,
    check_exists: bool = True,
    lock_fallback: bool = False,
) -> str:
    """Get filename for a conda yaml or pip requirements file."""

    if lock_fallback:
        try:
            return infer_requirement_path(
                name=name,
                ext=ext,
                python_version=python_version,
                lock=lock,
                check_exists=True,
                lock_fallback=False,
            )
        except FileNotFoundError:
            logger.info("Falling back to non-locked")
            return infer_requirement_path(
                name=name,
                ext=ext,
                python_version=python_version,
                lock=False,
                check_exists=True,
                lock_fallback=False,
            )

    if name is None:
        raise ValueError("must supply name")

    # adjust filename
    filename = name
    if ext is not None and not filename.endswith(ext):
        filename = filename + ext
    if python_version is not None:
        prefix = py_prefix(python_version)
        if not filename.startswith(prefix):
            filename = f"{prefix}-{filename}"

    if lock:
        if filename.endswith(".yaml"):
            filename = filename.rstrip(".yaml") + "-conda-lock.yml"
        elif filename.endswith(".yml"):
            filename = filename.rstrip(".yml") + "-conda-lock.yml"
        elif filename.endswith(".txt"):
            pass
        else:
            raise ValueError(f"unknown file extension for {filename}")

        filename = f"./requirements/lock/{filename}"
    else:
        filename = f"./requirements/{filename}"

    if check_exists:
        if not Path(filename).is_file():
            raise FileNotFoundError(f"{filename} does not exist")

    return filename


def _infer_requirement_paths(
    names: str | Iterable[str] | None,
    lock: bool = False,
    ext: str | None = None,
    python_version: str | None = None,
    lock_fallback: bool = False,
) -> list[str]:
    if names is None:
        return []
    elif isinstance(names, str):
        names = [names]

    return [
        infer_requirement_path(
            name,
            lock=lock,
            ext=ext,
            python_version=python_version,
            lock_fallback=lock_fallback,
        )
        for name in names
    ]


def is_conda_session(session: Session) -> bool:
    from nox.virtualenv import CondaEnv

    return isinstance(session.virtualenv, CondaEnv)


# * Main class ----------------------------------------------------------------
class Installer:
    """
    Class to handle installing package/dependecies

    Parameters
    ----------
    session : nox.Session
    update : bool, default=False
    lock : bool, default=False
    package: str, optional
    pip_deps : str or list of str, optional
        pip dependencies
    requirements : str or list of str
        pip requirement file(s) (pip install -r requirements[0] ...)
        Can either be a full path or a envname (for example,
        "test" will get resolved to ./requirements/test.txt)
    constraints : str or list of str
        pip constraint file(s) (pip install -c ...)
    config_path :
        Where to save env config for furture comparison.  Defaults to
        `session.create_tmp() / "env.json"`.
    """

    def __init__(
        self,
        session: Session,
        *,
        update: bool = False,
        lock: bool = False,
        package: str | bool | None = None,
        package_extras: str | Iterable[str] | None = None,
        pip_deps: str | Iterable[str] | None = None,
        requirements: PathLike | Iterable[PathLike] | None = None,
        constraints: PathLike | Iterable[PathLike] | None = None,
        config_path: PathLike | None = None,
        # conda specific things:
        conda_deps: str | Iterable[str] | None = None,
        conda_lock_path: PathLike | None = None,
        channels: str | Iterable[str] | None = None,
    ) -> None:
        self.session = session

        self.update = update
        self.lock = lock

        if config_path is None:
            config_path = Path(self.session.create_tmp()) / "env.json"
        else:
            config_path = Path(config_path)
        self.config_path = config_path

        if isinstance(package, bool):
            package = "." if package else None
        self.package = package
        self.package_extras = _to_list_of_str(package_extras)

        self.pip_deps: list[str] = sorted(_remove_whitespace_list(pip_deps or []))

        self.requirements = sorted(_verify_paths(requirements))
        self.constraints = sorted(_verify_paths(constraints))

        # conda stuff
        self.conda_deps: list[str] = sorted(_remove_whitespace_list(conda_deps or []))
        self.channels: list[str] = sorted(_remove_whitespace_list(channels or []))

        if not self.is_conda_session:
            if self.conda_deps or self.channels or conda_lock_path:
                raise ValueError("passing conda parameters to non conda session")

        if self.lock:
            if not self.is_conda_session:
                if (not self.requirements) or self.pip_deps or self.constraints:
                    raise ValueError("Can only pass requirements for locked virtualenv")

            else:
                if conda_lock_path is None:
                    raise ValueError("Must pass conda_lock_path")
                elif (
                    self.conda_deps
                    or self.channels
                    or self.pip_deps
                    or self.requirements
                    or self.constraints
                ):
                    raise ValueError(
                        "Can not pass conda_deps, channels, pip_deps, requirements, constraints if using conda-lock"
                    )

                conda_lock_path = _verify_path(conda_lock_path)

        self.conda_lock_path = conda_lock_path

    @cached_property
    def config(self) -> dict[str, Any]:
        """Dictionary of relavent info for this session"""
        out: dict[str, Any] = {}
        out["lock"] = self.lock

        for k in ["package", "package_extras", "pip_deps", "conda_deps", "channels"]:
            if v := getattr(self, k):
                out[k] = v

        # file hashses
        for k in ["requirements", "constraints", "conda_lock_path"]:
            if v := getattr(self, k):
                if isinstance(v, str):
                    v = [v]
                out[k] = {str(path): _get_file_hash(path) for path in v}

        return out

    def save_config(self) -> Self:
        import json

        # in case config path got clobbered
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        self.session.log(f"saving config to {self.config_path}")
        with open(self.config_path, "w") as f:
            json.dump(self.config, f)
        return self

    def log_session(self) -> Self:
        logfile = Path(self.session.create_tmp()) / "env_info.txt"
        self.session.log(f"writing environment log to {logfile}")

        with logfile.open("w") as f:
            self.session.run("python", "--version", stdout=f)

            if self.is_conda_session:
                self.session.run("conda", "list", stdout=f, external=True)
            else:
                self.session.run("pip", "list", stdout=f)

        return self

    @cached_property
    def previous_config(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {}
        else:
            import json

            with self.config_path.open("r") as f:
                return json.load(f)  # type: ignore

    # Interface
    @property
    def python_version(self) -> str:
        return cast(str, self.session.python)

    @cached_property
    def is_conda_session(self) -> bool:
        return is_conda_session(self.session)

    @cached_property
    def skip_install(self) -> bool:
        try:
            return self.session._runner.global_config.no_install and self.session._runner.venv._reused  # type: ignore
        except Exception:
            return False

    @cached_property
    def changed(self) -> bool:
        out = self.config != self.previous_config

        msg = "changed" if out else "unchanged"
        self.session.log(f"session {self.session.name} {msg}")

        return out

        # if (out := self.config != self.previous_config):
        #     for k in (self.config.keys() | self.previous_config.keys()):
        #         current = self.config.get(k, None)
        #         previous = self.previous_config.get(k, None)

        #         if current != previous:
        #             print(self.session.log(f"{k} current : {current}"))
        #             print(self.session.log(f"{k} previous: {previous}"))
        # return out

    # Smart runners
    def run_commands(
        self,
        commands: Iterable[str | Iterable[str]] | None,
        external: bool = True,
        **kwargs: Any,
    ) -> Self:
        if commands:
            kwargs.update(external=external)
            for opt in combine_list_list_str(commands):
                self.session.run(*opt, **kwargs)
        return self

    def set_ipykernel_display_name(
        self,
        display_name: str | None = None,
        update: bool = False,
    ) -> Self:
        if not display_name or (not self.is_conda_session) or self.skip_install:
            pass
        elif self.changed or update or self.update:
            command = f"python -m ipykernel install --sys-prefix --display-name {display_name}".split()
            # continue if fails
            self.session.run(*command, success_codes=[0, 1])

        return self

    @property
    def _package_str(self) -> str:
        if self.package is None:
            raise ValueError("Attempting to use unset package")

        if self.package_extras:
            package_extras = ",".join(self.package_extras)
            return f"{self.package}[{package_extras}]"
        else:
            return self.package

    def install_all(
        self,
        update_package: bool = False,
        log_session: bool = False,
        save_config: bool = True,
        display_name: str | None = None,
    ) -> Self:
        out = (
            (self.conda_install_deps() if self.is_conda_session else self)
            .pip_install_deps()
            .pip_install_package(update=update_package)
            .set_ipykernel_display_name(display_name=display_name)
        )

        if log_session:
            out = out.log_session()

        if save_config:
            out = out.save_config()

        return out

    def pip_install_package(
        self,
        *args: Any,
        update: bool = False,
        develop: bool = True,
        no_deps: bool = True,
        opts: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """Install the package"""
        if self.package is None or self.skip_install:
            pass

        elif self.changed or (update := self.update or update):
            command = [self._package_str]

            if develop:
                command.insert(0, "-e")

            if no_deps:
                command.append("--no-deps")

            if update:
                command.append("--upgrade")

            if opts:
                command.extend(combine_list_str(opts))

            self.session.install(*command, *args, **kwargs)

        return self

    def pip_install_deps(
        self,
        *args: Any,
        update: bool = False,
        opts: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        if self.skip_install:
            pass

        elif self.changed or (update := update or self.update):
            install_args: list[str] = (
                prepend_flag("-r", *self.requirements)
                + prepend_flag("-c", *self.constraints)
                + self.pip_deps
            )

            if install_args:
                if update:
                    install_args = ["--upgrade"] + install_args

                if opts:
                    install_args.extend(combine_list_str(opts))
                self.session.install(*install_args, *args, **kwargs)

        return self

    def conda_install_deps(
        self,
        *args: Any,
        update: bool = False,
        prune: bool = True,
        extras: str | Iterable[str] | None = None,
        opts: str | Iterable[str] | None = None,
        channels: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        if self.lock:
            return self._conda_install_lock(
                update=update, extras=extras, opts=opts, **kwargs
            )
        else:
            return self._conda_install_deps(
                update=update, opts=opts, channels=channels, prune=prune, **kwargs
            )

    def _conda_install_lock(
        self,
        update: bool = False,
        extras: str | Iterable[str] | None = None,
        opts: str | Iterable[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        if (not self.conda_lock_path) or self.skip_install:
            pass

        elif self.changed or (update := (update or self.update)):
            if extras:
                if isinstance(extras, str):
                    extras = extras.split(",")
                extras = prepend_flag("--extras", extras)
            else:
                extras = []

            if opts:
                opts = combine_list_str(opts)
            else:
                opts = []

            kwargs.setdefault("silent", True)
            kwargs["external"] = True

            if tmpdir := os.environ.get("TMPDIR"):
                kwargs["env"] = {"TMPDIR": tmpdir}

            self.session.run_always(
                "conda-lock",
                "install",
                "--mamba",
                *extras,
                *opts,
                "-p",
                str(self.session.virtualenv.location),
                str(self.conda_lock_path),
                **kwargs,
            )

        return self

    def _conda_install_deps(
        self,
        update: bool = False,
        opts: str | Iterable[str] | None = None,
        channels: str | Iterable[str] | None = None,
        prune: bool = True,
        **kwargs: Any,
    ) -> Self:
        if (not self.conda_deps) or self.skip_install:
            pass

        elif self.changed or (update := (update or self.update)):
            channels = channels or self.channels
            if channels:
                kwargs.update(channel=channels)

            deps = list(self.conda_deps)
            if update:
                deps.insert(0, "--update-all")

            if prune:
                deps.insert(0, "--prune")

            if opts:
                deps.extend(combine_list_str(opts))

            self.session.conda_install(*deps, **kwargs)

        return self

    # Super convenience methods
    @classmethod
    def from_yaml(
        cls,
        session: Session,
        envname: str | Iterable[str] | None = None,
        paths: PathLike | Iterable[PathLike] | None = None,
        conda_deps: str | Iterable[str] | None = None,
        pip_deps: str | Iterable[str] | None = None,
        channels: str | Iterable[str] | None = None,
        remove_python: bool = True,
        **kwargs: Any,
    ) -> Self:
        if paths or envname:
            assert isinstance(session.python, str)

            paths = _infer_requirement_paths(
                envname, ext=".yaml", lock=False, python_version=session.python
            ) + _verify_paths(paths)

            session.log(f"Using yaml files: {paths}")
            channels, conda_deps, pip_deps, _ = parse_envs(
                *paths,
                remove_python=remove_python,
                deps=conda_deps,
                reqs=pip_deps,
                channels=channels,
            )

        return cls(
            session=session,
            conda_deps=conda_deps,
            pip_deps=pip_deps,
            channels=channels,
            **kwargs,
        )

    @classmethod
    def _from_envname_pip(
        cls,
        session: Session,
        envname: str | Iterable[str] | None = None,
        lock: bool = False,
        lock_fallback: bool = False,
        requirements: PathLike | Iterable[PathLike] | None = None,
        **kwargs: Any,
    ) -> Self:
        if lock:
            assert isinstance(session.python, str)
            requirements = _verify_paths(requirements) + _infer_requirement_paths(
                envname,
                ext=".txt",
                lock=lock,
                python_version=session.python,
                lock_fallback=lock_fallback,
            )

        else:
            if envname is not None:
                requirements = _verify_paths(requirements) + _infer_requirement_paths(
                    envname, ext=".txt"
                )

        return cls(
            session=session,
            lock=lock,
            requirements=requirements,
            **kwargs,
        )

    @classmethod
    def _from_envname_conda(
        cls,
        session: Session,
        envname: str | Iterable[str] | None = None,
        lock: bool = False,
        lock_fallback: bool = True,
        conda_lock_path: PathLike | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Parameters
        ----------
        envname :
            Base name for file.  For example, passing
            envname = "dev" will convert to
            `requirements/py{py}-dev.yaml` for `filename`
        """

        if lock:
            # get files from envname

            if conda_lock_path is None:
                assert isinstance(
                    envname, str
                ), "Must supply conda_lock_path or envname"
                conda_lock_path = infer_requirement_path(envname, ext=".yaml", python_version=session.python, lock=lock, lock_fallback=lock_fallback)  # type: ignore

            return cls(
                session=session, lock=lock, conda_lock_path=conda_lock_path, **kwargs
            )

        else:
            return cls.from_yaml(session=session, envname=envname, lock=lock, **kwargs)

    @classmethod
    def from_envname(
        cls,
        session: Session,
        envname: str | Iterable[str] | None = None,
        lock: bool = False,
        lock_fallback: bool | None = None,
        **kwargs: Any,
    ) -> Self:
        if lock_fallback is None:
            lock_fallback = is_conda_session(session)

        # if lock_fallback and lock:
        #     try:
        #         return cls.from_envname(
        #             session=session,
        #             envname=envname,
        #             lock=lock,
        #             lock_fallback=False,
        #             **kwargs,
        #         )
        #     except Exception:
        #         session.log("Falling back to non-locked")
        #         return cls.from_envname(
        #             session=session,
        #             envname=envname,
        #             lock=False,
        #             lock_fallback=False,
        #             **kwargs,
        #         )

        if is_conda_session(session):
            return cls._from_envname_conda(
                session=session,
                envname=envname,
                lock=lock,
                lock_fallback=lock_fallback,
                **kwargs,
            )
        else:
            return cls._from_envname_pip(
                session=session,
                envname=envname,
                lock=lock,
                lock_fallback=lock_fallback,
                **kwargs,
            )


# * Utilities --------------------------------------------------------------------------
def _to_list_of_str(x: str | Iterable[str] | None) -> list[str]:
    if x is None:
        return []
    elif isinstance(x, str):
        return [x]
    elif isinstance(x, list):
        return x
    else:
        return list(x)


def _remove_whitespace(s: str) -> str:
    import re

    return re.sub(r"\s+", "", s)


def _remove_whitespace_list(s: str | Iterable[str]) -> list[str]:
    if isinstance(s, str):
        s = [s]
    return [_remove_whitespace(x) for x in s]


def combine_list_str(opts: str | Iterable[str]) -> list[str]:
    if isinstance(opts, str):
        opts = [opts]

    if opts:
        return shlex.split(" ".join(opts))
    else:
        return []


def combine_list_list_str(opts: Iterable[str | Iterable[str]]) -> Iterable[list[str]]:
    return (combine_list_str(opt) for opt in opts)


def sort_like(values: Iterable[Any], like: Iterable[Any]) -> list[Any]:
    """Sort `values` in order of `like`."""
    # only unique
    sorter = {k: i for i, k in enumerate(like)}
    return sorted(set(values), key=lambda k: sorter[k])


def update_target(
    target: str | Path, *deps: str | Path, allow_missing: bool = False
) -> bool:
    """Check if target is older than deps:"""
    target = Path(target)

    deps_filtered: list[Path] = []
    for d in map(Path, deps):
        if d.exists():
            deps_filtered.append(d)
        elif not allow_missing:
            raise ValueError(f"dependency {d} does not exist")

    if not target.exists():
        return True
    else:
        target_time = target.stat().st_mtime

        update = any(target_time < dep.stat().st_mtime for dep in deps_filtered)

    return update


def prepend_flag(flag: str, *args: str | Iterable[str]) -> list[str]:
    """
    Add in a flag before each arg.

    >>> prepent_flag("-k", "a", "b")
    ["-k", "a", "-k", "b"]
    """

    args_: list[str] = []
    for x in args:
        if isinstance(x, str):
            args_.append(x)
        else:
            args_.extend(x)

    return sum([[flag, _] for _ in args_], [])  # pyright: ignore


def _get_file_hash(path: str | Path, buff_size: int = 65536) -> str:
    import hashlib

    md5 = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            data = f.read(buff_size)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


@contextmanager
def check_for_change_manager(
    *deps: str | Path,
    hash_path: str | Path | None = None,
    target_path: str | Path | None = None,
    force_write: bool = False,
):
    """
    Context manager to look for changes in dependencies.

    Yields
    ------
    changed: bool

    If exit normaly, write hashes to hash_path file
    """

    try:
        changed, hashes, hash_path = check_hash_path_for_change(
            *deps, target_path=target_path, hash_path=hash_path
        )

        yield changed

    except Exception as e:
        raise e

    else:
        if force_write or changed:
            logger.info(f"Writing {hash_path}")

            # make sure the parent directory exists:
            hash_path.parent.mkdir(parents=True, exist_ok=True)
            write_hashes(hash_path=hash_path, hashes=hashes)


def check_hash_path_for_change(
    *deps: str | Path,
    target_path: str | Path | None = None,
    hash_path: str | Path | None = None,
) -> tuple[bool, dict[str, str], Path]:
    """
    Checks a json file `hash_path` for hashes of `other_paths`.

    if specify target_path and no hash_path, set `hash_path=target_path.parent / (target_path.name + ".hash.json")`.
    if specify hash_path and no target, set

    Parameters
    ----------
    *deps :
        files on which target_path/hash_path depends.
    hash_path :
        Path of file containing hashes of `deps`.
    target_path :
        Target file (i.e., the final file to be created).
        Defaults to hash_path.


    Returns
    -------
    changed : bool
    hashes : dict[str, str]
    hash_path : Path

    """
    import json

    msg = "Must specify target_path or hash_path"

    if target_path is None:
        assert hash_path is not None, msg
        target_path = hash_path = Path(hash_path)
    else:
        target_path = Path(target_path)
        if hash_path is None:
            hash_path = target_path.parent / (target_path.name + ".hash.json")
        else:
            hash_path = Path(hash_path)

    hashes: dict[str, str] = {
        os.path.relpath(k, hash_path.parent): _get_file_hash(k) for k in deps
    }

    if not target_path.is_file():
        changed = True

    elif hash_path.is_file():
        with open(hash_path) as f:
            previous_hashes: dict[str, Any] = json.load(f)

        changed = False
        for k, h in hashes.items():
            previous = previous_hashes.get(k)
            if previous is None or previous != h:
                changed = True
                hashes = {**previous_hashes, **hashes}
                break

    else:
        changed = True

    return changed, hashes, hash_path


def write_hashes(hash_path: str | Path, hashes: dict[str, Any]) -> None:
    import json

    with open(hash_path, "w") as f:
        json.dump(hashes, f, indent=2)
        f.write("\n")


def open_webpage(path: str | Path | None = None, url: str | None = None) -> None:
    """
    Open webpage from path or url.

    Useful if want to view webpage with javascript, etc., as well as static html.
    """
    import webbrowser
    from urllib.request import pathname2url

    if path:
        url = "file://" + pathname2url(str(Path(path).absolute()))
    if url:
        webbrowser.open(url)


def session_run_commands(
    session: Session,
    commands: list[list[str]] | None,
    external: bool = True,
    **kws: Any,
) -> None:
    """Run commands command."""

    if commands:
        kws.update(external=external)
        for opt in combine_list_list_str(commands):
            session.run(*opt, **kws)


# ** Conda -----------------------------------------------------------------------------
def parse_envs(
    *paths: str | Path,
    remove_python: bool = True,
    deps: str | Iterable[str] | None = None,
    reqs: str | Iterable[str] | None = None,
    channels: str | Iterable[str] | None = None,
) -> tuple[set[str], set[str], set[str], str | None]:
    """Parse an `environment.yaml` file."""
    import re

    from ruamel.yaml import YAML

    def _default(x: str | Iterable[str] | None) -> set[str]:
        if x is None:
            return set()
        elif isinstance(x, str):
            x = [x]
        return set(x)

    channels = _default(channels)
    deps = _default(deps)
    reqs = _default(reqs)
    name = None

    python_match = re.compile(r"\s*(python)\s*[~<=>].*")

    def _get_context(path: str | Path | TextIO) -> TextIO | Path:
        if isinstance(path, str):
            return open(path)
        elif isinstance(path, Path):
            return path
        else:
            from contextlib import nullcontext

            return nullcontext(path)  # type: ignore

    for path in paths:
        with _get_context(path) as f:
            data = YAML(typ="safe", pure=True).load(f)

        channels.update(data.get("channels", []))
        name = data.get("name", name)

        # check dependencies for pip
        for d in data.get("dependencies", []):
            if isinstance(d, dict):
                reqs.update(cast(list[str], d.get("pip")))  # pyright: ignore
            else:
                if remove_python and not python_match.match(d):
                    deps.add(d)

    return channels, deps, reqs, name


# * User config ------------------------------------------------------------------------
def load_nox_config(path: str | Path = "./config/userconfig.toml") -> dict[str, Any]:
    """
    Load user toml config file.

    File should look something like:

    [nox.python]
    paths = ["~/.conda/envs/python-3.*/bin"]

    # Extras for environments
    # for example, could have
    # dev = ["dev", "nox", "tools"]
    [nox.extras]
    dev = ["dev", "nox"]
    """

    from .projectconfig import ProjectConfig

    return ProjectConfig.from_path_and_environ(path).to_nox_config()
