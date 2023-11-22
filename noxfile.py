"""Config file for nox."""
# * Imports ----------------------------------------------------------------------------
from __future__ import annotations

import shutil
import sys

# Should only use on python version > 3.10
assert sys.version_info >= (3, 10)

from dataclasses import replace  # noqa
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Sequence,
    TypeVar,
    cast,
    overload,
)

if sys.version_info > (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if sys.version_info > (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated


import nox  # type: ignore[unused-ignore,import]
from noxopt import NoxOpt, Option, Session  # type: ignore[unused-ignore,import]

# fmt: off
sys.path.insert(0, ".")

from tools.noxtools import (
    Installer,
    combine_list_str,
    is_conda_session,
    load_nox_config,
    open_webpage,
    prepend_flag,
    session_run_commands,
    sort_like,
    update_target,
)

sys.path.pop(0)
# fmt: on


# * Names ------------------------------------------------------------------------------

PACKAGE_NAME = "open-notebook"
IMPORT_NAME = "open_notebook"
KERNEL_BASE = "open_notebook"

# * nox options ------------------------------------------------------------------------

ROOT = Path(__file__).parent

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["test"]
# Using ".nox/{project-name}/envs" instead of ".nox" to store environments.
# This fixes problems with ipykernel/nb_conda_kernel and some other dev tools
# that expect conda environments to be in something like ".../a/path/miniforge/envs/env".
nox.options.envdir = f".nox/{PACKAGE_NAME}/envs"

# * User Config ------------------------------------------------------------------------

CONFIG = load_nox_config()

# * Options ----------------------------------------------------------------------------

PYTHON_ALL_VERSIONS = ["3.8", "3.9", "3.10", "3.11", "3.12"]
PYTHON_DEFAULT_VERSION = "3.11"

# conda/mamba
if shutil.which("mamba"):
    CONDA_BACKEND = "mamba"
elif shutil.which("conda"):
    CONDA_BACKEND = "conda"  # pyright: ignore
else:
    raise ValueError("neither conda or mamba found")

CONDA_DEFAULT_KWS = {"python": PYTHON_DEFAULT_VERSION, "venv_backend": CONDA_BACKEND}
CONDA_ALL_KWS = {"python": PYTHON_ALL_VERSIONS, "venv_backend": CONDA_BACKEND}

DEFAULT_KWS = {"python": PYTHON_DEFAULT_VERSION}
ALL_KWS = {"python": PYTHON_ALL_VERSIONS}


# * noxopt -----------------------------------------------------------------------------
group = NoxOpt(auto_tag=True)

F = TypeVar("F", bound=Callable[..., Any])
C: TypeAlias = Callable[[F], F]


@overload
def group_session(func: F, **kwargs: Any) -> F:
    ...


@overload
def group_session(**kwargs: Any) -> Callable[[F], F]:
    ...


def group_session(func: F | None, **kwargs: Any) -> F | Callable[[F], F]:
    return group.session(func, **kwargs)  # type: ignore


# CONDA_DEFAULT_SESSION = group_session(None, **CONDA_DEFAULT_KWS)
# CONDA_ALL_SESSION = group_session(None, **CONDA_ALL_KWS)

# DEFAULT_SESSION = group_session(None, **DEFAULT_KWS)
# ALL_SESSION = group_session(None, **)


CONDA_DEFAULT_SESSION = cast(C[F], group.session(**CONDA_DEFAULT_KWS))  # type: ignore
CONDA_ALL_SESSION = cast(C[F], group.session(**CONDA_ALL_KWS))  # type: ignore

DEFAULT_SESSION = cast(C[F], group.session(python=PYTHON_DEFAULT_VERSION))  # type: ignore
ALL_SESSION = cast(C[F], group.session(python=PYTHON_ALL_VERSIONS))  # type: ignore

NOPYTHON_SESSION = cast(C[F], group.session(python=False))  # type: ignore
INHERITED_SESSION = cast(C[F], group.session)  # type: ignore

OPTS_OPT = Option(nargs="*", type=str)
RUN_OPT = Option(
    nargs="*",
    type=str,
    action="append",
    help="run passed command_demo using `external=True` flag",
)

CMD_OPT = Option(nargs="*", type=str, help="cmd to be run")
LOCK_OPT = Option(type=bool, help="If True, use conda-lock")


OPT_TYPE: TypeAlias = list[str] | None
CMD_TYPE: TypeAlias = list[str] | None
RUN_TYPE: TypeAlias = list[list[str]] | None


def opts_replace(**kwargs: Any) -> Option:
    return replace(OPTS_OPT, **kwargs)


def cmd_replace(**kwargs: Any) -> Option:
    return replace(CMD_OPT, **kwargs)


def run_replace(**kwargs: Any) -> Option:
    return replace(RUN_OPT, **kwargs)


LOCK_CLI = Annotated[bool, LOCK_OPT]
RUN_CLI = Annotated[RUN_TYPE, RUN_OPT]
RUN_INTERNAL_CLI = Annotated[
    RUN_TYPE, run_replace(help="run arbitrary internal command")
]

TEST_OPTS_CLI = Annotated[
    OPT_TYPE, opts_replace(help="extra arguments/flags to pytest")
]
EXTRAS_CLI = Annotated[OPT_TYPE, cmd_replace(help="extras to include from package")]
PYTHON_PATHS_CLI = Annotated[
    OPT_TYPE, cmd_replace(help="python paths to append to PATHS")
]

UPDATE_CLI = Annotated[
    bool,
    Option(
        type=bool,
        help="If True, force update of installed packages",
        flags=("--update", "-U"),
    ),
]

UPDATE_PACKAGE_CLI = Annotated[
    bool,
    Option(
        type=bool,
        help="If True, and session uses package, reinstall package",
        flags=("--update-package", "-P"),
    ),
]

VERSION_CLI = Annotated[
    str, Option(type=str, help="Version to substitute or check against")
]

LOG_SESSION_CLI = Annotated[
    bool,
    Option(
        type=bool,
        help="If flag included, log python and package (if installed) version",
    ),
]


DOC_CMD_CLI = Annotated[
    CMD_TYPE,
    cmd_replace(
        choices=[
            "html",
            "build",
            "symlink",
            "clean",
            "livehtml",
            "linkcheck",
            "spelling",
            "showlinks",
            "release",
            "open",
            "serve",
        ],
        flags=("--docs-cmd", "-d"),
    ),
]


DIST_PYPI_CMD_CLI = Annotated[
    CMD_TYPE,
    cmd_replace(
        choices=["clean", "build", "testrelease", "release"],
        flags=("--dist-pypi-cmd", "-p"),
    ),
]


# * Environments------------------------------------------------------------------------
# ** Dev (conda)


@nox.session(name="example-venv", **DEFAULT_KWS)
@nox.session(**CONDA_DEFAULT_KWS)
def example(session: Session, hello: str = "hello") -> None:
    print("hello:", hello)
    print(session.name)
    print(type(session.virtualenv))


# example_venv = group_session(example, name="example-venv", **DEFAULT_KWS)
# example = CONDA_DEFAULT_SESSION(example)


def dev(
    session: Session,
    dev_run: RUN_CLI = None,
    lock: LOCK_CLI = False,
    update: UPDATE_CLI = False,
    update_package: UPDATE_PACKAGE_CLI = False,
    log_session: bool = False,
) -> None:
    """Create dev env using conda."""
    # using conda

    (
        Installer.from_envname(
            session=session, envname="dev", lock=lock, update=update, package="."
        )
        .install_all(
            update_package=update_package,
            log_session=log_session,
            display_name=f"{PACKAGE_NAME}-{session.name}",
        )
        .run_commands(dev_run)
    )


dev_venv = group_session(func=dev, name="dev-venv", **DEFAULT_KWS)
dev = group_session(func=dev, **CONDA_DEFAULT_KWS)


# ** bootstrap
@NOPYTHON_SESSION
def bootstrap(session: Session) -> None:
    """Run config, reqs, and dev"""

    session.notify("config")
    session.notify("requirements")
    session.notify("dev")


# ** config
@NOPYTHON_SESSION
def config(
    session: Session,
    dev_extras: EXTRAS_CLI = None,
    python_paths: PYTHON_PATHS_CLI = None,
) -> None:
    """Create the file ./config/userconfig.toml"""

    args: list[str] = []
    if dev_extras:
        args += ["--dev-extras"] + dev_extras
    if python_paths:
        args += ["--python-paths"] + python_paths

    session.run("python", "tools/projectconfig.py", *args)


# ** requirements
@NOPYTHON_SESSION
def pyproject2conda(
    session: Session,
    update: UPDATE_CLI = False,
) -> None:
    """Alias to reqs"""
    session.notify("requirements")


@INHERITED_SESSION
def requirements(
    session: Session,
    update: UPDATE_CLI = False,
    requirements_force: bool = False,
    log_session: bool = False,
) -> None:
    """
    Create environment.yaml and requirement.txt files from pyproject.toml using pyproject2conda.

    These will be placed in the directory "./environments".
    """

    (
        Installer(
            session=session,
            pip_deps="pyproject2conda>=0.8.0",
            update=update,
        ).install_all(log_session=log_session)
    )

    session.run(
        "pyproject2conda",
        "project",
        "--verbose",
        *(["--overwrite", "force"] if requirements_force else []),
    )


# ** conda-lock
@DEFAULT_SESSION
def conda_lock(
    session: Session,
    update: UPDATE_CLI = False,
    conda_lock_channel: Annotated[
        OPT_TYPE, opts_replace(help="conda channels to use")
    ] = None,
    conda_lock_platform: Annotated[
        OPT_TYPE,
        opts_replace(
            help="platforms to build lock files for",
            choices=["osx-64", "linux-64", "win-64", "all"],
        ),
    ] = None,
    conda_lock_include: Annotated[
        OPT_TYPE, opts_replace(help="lock files to create")
    ] = None,
    conda_lock_run: RUN_CLI = None,  # noqa
    conda_lock_mamba: bool = False,
    conda_lock_force: bool = False,
) -> None:
    """Create lock files using conda-lock."""

    (
        Installer(
            session=session,
            pip_deps="conda-lock>=2.2.0",
            update=update,
        ).install_all()
    )

    session.run("conda-lock", "--version")

    conda_lock_exclude = ["test-extras"]

    platform = conda_lock_platform
    if not platform:
        platform = ["osx-64"]
    elif "all" in platform:
        platform = ["linux-64", "osx-64", "win-64"]

    channel = conda_lock_channel
    if not channel:
        channel = ["conda-forge"]

    def create_lock(path: Path) -> None:
        name = path.with_suffix("").name
        lockfile = path.parent / "lock" / f"{name}-conda-lock.yml"
        deps = [str(path)]

        # check if skip
        env = "-".join(name.split("-")[1:])
        if conda_lock_include:
            if not any(c == env for c in conda_lock_include):  # pyright: ignore
                session.log(f"Skipping {lockfile} (include)")
                return

        if conda_lock_exclude:
            if any(c == env for c in conda_lock_exclude):  # pyright: ignore
                session.log(f"Skipping {lockfile} (exclude)")
                return

        if conda_lock_force or update_target(lockfile, *deps):
            session.log(f"Creating {lockfile}")
            # insert -f for each arg
            if lockfile.exists():
                lockfile.unlink()
            session.run(
                "conda-lock",
                "--mamba" if conda_lock_mamba else "--no-mamba",
                *prepend_flag("-c", *channel),
                *prepend_flag("-p", *platform),
                *prepend_flag("-f", *deps),
                f"--lockfile={lockfile}",
            )
        else:
            session.log(f"Skipping {lockfile} (exists)")

    session_run_commands(session, conda_lock_run)
    for path in (ROOT / "requirements").relative_to(ROOT.cwd()).glob("py*.yaml"):
        create_lock(path)


# ** testing
def _test(
    session: nox.Session,
    run: RUN_TYPE,
    test_no_pytest: bool,
    test_opts: OPT_TYPE,
    no_cov: bool,
) -> None:
    import os

    tmpdir = os.environ.get("TMPDIR", None)

    session_run_commands(session, run)
    if not test_no_pytest:
        opts = combine_list_str(test_opts or [])
        if not no_cov:
            session.env["COVERAGE_FILE"] = str(Path(session.create_tmp()) / ".coverage")

            if "--cov" not in opts:
                opts.append("--cov")

        # Because we are testing if temporary folders
        # have git or not, we have to make sure we're above the
        # not under this repo
        # so revert to using the top level `TMPDIR`
        if tmpdir:
            session.env["TMPDIR"] = tmpdir

        session.run("pytest", *opts)


def test(
    session: Session,
    test_no_pytest: bool = False,
    test_opts: TEST_OPTS_CLI = None,
    test_run: RUN_CLI = None,
    lock: LOCK_CLI = False,
    update: UPDATE_CLI = False,
    update_package: UPDATE_PACKAGE_CLI = False,
    log_session: bool = False,
    no_cov: bool = False,
) -> None:
    """Test environments with conda installs."""
    (
        Installer.from_envname(
            session=session,
            envname="test",
            lock=lock,
            package=".",
            update=update,
        ).install_all(log_session=log_session, update_package=update_package)
    )

    _test(
        session=session,
        run=test_run,
        test_no_pytest=test_no_pytest,
        test_opts=test_opts,
        no_cov=no_cov,
    )


test_venv = group_session(test, name="test-venv", **ALL_KWS)
test = group_session(test, name="test", **CONDA_ALL_KWS)


# ** coverage
@INHERITED_SESSION
def coverage(
    session: Session,
    coverage_cmd: Annotated[
        CMD_TYPE, cmd_replace(choices=["erase", "combine", "report", "html", "open"])
    ] = None,
    coverage_run: RUN_CLI = None,
    coverage_run_internal: RUN_CLI = None,
    update: UPDATE_CLI = False,
) -> None:
    runner = Installer(
        session=session, pip_deps="coverage[toml]", update=update
    ).install_all()

    runner.run_commands(coverage_run)

    cmd = coverage_cmd or []
    if not cmd and not coverage_run and not coverage_run_internal:
        cmd = ["combine", "html", "report"]

    session.log(f"{cmd}")

    for c in cmd:
        if c == "combine":
            paths = list(
                Path(session.virtualenv.location).parent.glob("test-*/tmp/.coverage")
            )
            if update_target(".coverage", *paths):
                session.run("coverage", "combine", "--keep", "-a", *map(str, paths))
        elif c == "open":
            open_webpage(path="htmlcov/index.html")
        else:
            session.run("coverage", c)

    runner.run_commands(coverage_run_internal)


# ** Docs
def docs(
    session: nox.Session,
    docs_cmd: DOC_CMD_CLI = None,
    docs_run: RUN_CLI = None,
    lock: LOCK_CLI = False,
    update: UPDATE_CLI = False,
    update_package: UPDATE_PACKAGE_CLI = False,
    version: VERSION_CLI = "",
    log_session: bool = False,
) -> None:
    """
    Runs make in docs directory. For example, 'nox -s docs -- --docs-cmd
    html' -> 'make -C docs html'. With 'release' option, you can set the
    message with 'message=...' in posargs.
    """

    runner = Installer.from_envname(
        session=session, envname="docs", lock=lock, package=".", update=update
    ).install_all(
        update_package=update_package,
        log_session=log_session,
        display_name=f"{PACKAGE_NAME}-docs",
    )

    # _docs(session=session, cmd=docs_cmd, run=docs_run, version=version)

    if version:
        session.env["SETUPTOOLS_SCM_PRETEND_VERSION"] = version

    runner.run_commands(docs_run)

    cmd = docs_cmd or []
    if not docs_run and not cmd:
        cmd = ["html"]

    if "symlink" in cmd:
        cmd.remove("symlink")
        _create_doc_examples_symlinks(session)

    if open_page := "open" in cmd:
        cmd.remove("open")

    if serve := "serve" in cmd:
        open_webpage(url="http://localhost:8000")
        cmd.remove("serve")

    if cmd:
        args = ["make", "-C", "docs"] + combine_list_str(cmd)
        session.run(*args, external=True)

    if open_page:
        open_webpage(path="./docs/_build/html/index.html")

    if serve and "livehtml" not in cmd:
        session.run(
            "python",
            "-m",
            "http.server",
            "-d",
            "docs/_build/html",
            "-b",
            "127.0.0.1",
            "8000",
        )


docs_venv = group_session(docs, name="docs-venv", **DEFAULT_KWS)
docs = group_session(docs, name="docs", **CONDA_DEFAULT_KWS)


# ** Dist pypi
@DEFAULT_SESSION
def dist_pypi(
    session: nox.Session,
    dist_pypi_run: RUN_CLI = None,
    dist_pypi_cmd: DIST_PYPI_CMD_CLI = None,
    update: UPDATE_CLI = False,
    version: VERSION_CLI = "",
    log_session: bool = False,
) -> None:
    """Run 'nox -s dist-pypi -- {clean, build, testrelease, release}'."""

    runner = Installer.from_envname(
        session=session,
        envname="dist-pypi",
        update=update,
    ).install_all(log_session=log_session)

    if version:
        session.env["SETUPTOOLS_SCM_PRETEND_VERSION"] = version

    runner.run_commands(dist_pypi_run)

    cmd = dist_pypi_cmd or []
    if not dist_pypi_run and not cmd:
        cmd = ["build"]

    if cmd:
        if "build" in cmd:
            cmd.append("clean")
        cmd = sort_like(cmd, ["clean", "build", "testrelease", "release"])

        session.log(f"cmd={cmd}")

        for command in cmd:
            if command == "clean":
                session.run("rm", "-rf", "dist", external=True)
            elif command == "build":
                session.run("python", "-m", "build", "--outdir", "dist/")

            elif command == "testrelease":
                session.run("twine", "upload", "--repository", "testpypi", "dist/*")

            elif command == "release":
                session.run("twine", "upload", "dist/*")


# ** Dist conda
@CONDA_DEFAULT_SESSION
def dist_conda(
    session: nox.Session,
    dist_conda_run: RUN_CLI = None,
    dist_conda_cmd: Annotated[
        CMD_TYPE,
        cmd_replace(
            choices=[
                "recipe",
                "build",
                "clean",
                "clean-recipe",
                "clean-build",
                "recipe-cat-full",
            ],
            flags=("--dist-conda-cmd", "-c"),
        ),
    ] = None,
    sdist_path: str = "",
    update: UPDATE_CLI = False,
    log_session: bool = False,
    version: VERSION_CLI = "",
) -> None:
    """Runs make -C dist-conda posargs."""

    runner = Installer.from_envname(
        session=session, envname="dist-conda", update=update
    ).install_all(log_session=log_session)

    run, cmd = dist_conda_run, dist_conda_cmd
    runner.run_commands(run)

    if not run and not cmd:
        cmd = ["recipe"]

    # make directory?
    if not (d := Path("./dist-conda")).exists():
        d.mkdir()

    if cmd:
        if "recipe" in cmd:
            cmd.append("clean-recipe")
        if "build" in cmd:
            cmd.append("clean-build")
        if "clean" in cmd:
            cmd.extend(["clean-recipe", "clean-build"])
            cmd.remove("clean")

        cmd = sort_like(
            cmd, ["recipe-cat-full", "clean-recipe", "recipe", "clean-build", "build"]
        )

        if not sdist_path:
            sdist_path = PACKAGE_NAME
            if version:
                sdist_path = f"{sdist_path}=={version}"

        for command in cmd:
            if command == "clean-recipe":
                session.run("rm", "-rf", f"dist-conda/{PACKAGE_NAME}", external=True)
            elif command == "clean-build":
                session.run("rm", "-rf", "dist-conda/build", external=True)
            elif command == "recipe":
                session.run(
                    "grayskull",
                    "pypi",
                    sdist_path,
                    "--sections",
                    "package",
                    "source",
                    "build",
                    "requirements",
                    "-o",
                    "dist-conda",
                )
                _append_recipe(
                    f"dist-conda/{PACKAGE_NAME}/meta.yaml", "config/recipe-append.yaml"
                )
                session.run(
                    "cat", f"dist-conda/{PACKAGE_NAME}/meta.yaml", external=True
                )
            elif command == "recipe-cat-full":
                import tempfile

                with tempfile.TemporaryDirectory() as d:  # type: ignore[assignment,unused-ignore]
                    session.run(
                        "grayskull",
                        "pypi",
                        sdist_path,
                        "-o",
                        str(d),
                    )
                    session.run(
                        "cat", str(Path(d) / PACKAGE_NAME / "meta.yaml"), external=True
                    )

            elif command == "build":
                session.run(
                    "conda",
                    "mambabuild",
                    "--output-folder=dist-conda/build",
                    "--no-anaconda-upload",
                    "dist-conda",
                )


# ** lint
@INHERITED_SESSION
def lint(
    session: nox.Session,
    lint_run: RUN_CLI = None,
    update: UPDATE_CLI = False,
    log_session: bool = False,
) -> None:
    """
    Run linters with pre-commit.

    Defaults to `pre-commit run --all-files`.
    To run something else pass, e.g.,
    `nox -s lint -- --lint-run "pre-commit run --hook-stage manual --all-files`
    """

    runner = Installer(
        session=session, pip_deps="pre-commit", update=update
    ).install_all(log_session=log_session)

    if lint_run:
        runner.run_commands(lint_run, external=True)
    else:
        session.run("pre-commit", "run", "--all-files")


# ** type checking
def typing(
    session: nox.Session,
    typing_cmd: Annotated[
        CMD_TYPE,
        cmd_replace(
            choices=[
                "clean",
                "mypy",
                "pyright",
                "pytype",
                "all",
                "nbqa-mypy",
                "nbqa-pyright",
                "nbqa-typing",
            ],
            flags=("--typing-cmd", "-m"),
        ),
    ] = None,
    typing_run: RUN_CLI = None,
    typing_run_internal: RUN_INTERNAL_CLI = None,
    lock: LOCK_CLI = False,
    update: UPDATE_CLI = False,
    log_session: bool = False,
) -> None:
    """Run type checkers (mypy, pyright, pytype)."""

    runner = (
        Installer.from_envname(
            session=session,
            envname="typing",
            lock=lock,
            update=update,
        )
        .install_all(log_session=log_session)
        .run_commands(typing_run)
    )

    cmd = typing_cmd or []
    if not typing_run and not typing_run_internal and not cmd:
        cmd = ["mypy", "pyright"]

    if "all" in cmd:
        cmd = ["mypy", "pyright", "pytype"]

    # set the cache directory for mypy
    session.env["MYPY_CACHE_DIR"] = str(Path(session.create_tmp()) / ".mypy_cache")

    def _run_info(cmd: str) -> None:
        session.run("which", cmd, external=True)
        session.run(cmd, "--version", external=True)

    if "clean" in cmd:
        cmd = [x for x in cmd if x != "clean"]

        for name in [".mypy_cache", ".pytype"]:
            p = Path(session.create_tmp()) / name
            if p.exists():
                session.log(f"removing cache {p}")
                shutil.rmtree(str(p))

    for c in cmd:
        if not c.startswith("nbqa"):
            _run_info(c)
        if c == "mypy":
            session.run("mypy", "--color-output")
        elif c == "pyright":
            session.run("pyright", external=True)
        elif c == "pytype":
            session.run("pytype", "-o", str(Path(session.create_tmp()) / ".pytype"))
        elif c.startswith("nbqa"):
            session.run("make", c, external=True)
        else:
            session.log(f"skipping unknown command {c}")

    runner.run_commands(typing_run_internal, external=False)


typing_venv = group_session(typing, name="typing-venv", **ALL_KWS)
typing = group_session(typing, name="typing", **CONDA_ALL_KWS)


# ** testdist (conda)
def _testdist(
    session: Session,
    test_no_pytest: bool = False,
    test_opts: TEST_OPTS_CLI = None,
    testdist_run: RUN_CLI = None,
    update: UPDATE_CLI = False,
    version: VERSION_CLI = "",
    log_session: bool = False,
) -> None:
    """Test conda distribution."""

    install_str = PACKAGE_NAME
    if version:
        install_str = f"{install_str}=={version}"

    if is_conda_session(session):
        pip_deps, conda_deps = None, install_str
    else:
        pip_deps, conda_deps = install_str, None

    (
        Installer.from_envname(
            session=session,
            envname="test-extras",
            conda_deps=conda_deps,
            pip_deps=pip_deps,
            update=update,
            channels="conda-forge",
        ).install_all(log_session=log_session)
    )

    _test(
        session=session,
        run=testdist_run,
        test_no_pytest=test_no_pytest,
        test_opts=test_opts,
        no_cov=True,
    )


testdist_pypi = group_session(_testdist, name="testdist-pypi", **ALL_KWS)
testdist_conda = group_session(_testdist, name="testdist-conda", **CONDA_ALL_KWS)


# * Utilities --------------------------------------------------------------------------
def _create_doc_examples_symlinks(session: nox.Session, clean: bool = True) -> None:
    """Create symlinks from docs/examples/*.md files to /examples/usage/..."""

    import os

    def usage_paths(path: Path) -> Iterator[Path]:
        with path.open("r") as f:
            for line in f:
                if line.startswith("usage/"):
                    yield Path(line.strip())

    def get_target_path(
        usage_path: str | Path,
        prefix_dir: str | Path = "./examples",
        exts: Sequence[str] = (".md", ".ipynb"),
    ) -> Path:
        path = Path(prefix_dir) / Path(usage_path)

        assert all(ext.startswith(".") for ext in exts)

        if path.exists():
            return path
        else:
            for ext in exts:
                p = path.with_suffix(ext)
                if p.exists():
                    return p

        raise ValueError(f"no path found for base {path}")

    root = Path("./docs/examples/")
    if clean:
        import shutil

        shutil.rmtree(root / "usage", ignore_errors=True)

    # get all md files
    paths = list(root.glob("*.md"))

    # read usage lines
    for path in paths:
        for usage_path in usage_paths(path):
            target = get_target_path(usage_path)
            link = root / usage_path.parent / target.name

            if link.exists():
                link.unlink()

            link.parent.mkdir(parents=True, exist_ok=True)

            target_rel = os.path.relpath(target, start=link.parent)
            session.log(f"linking {target_rel} -> {link}")

            os.symlink(target_rel, link)


def _append_recipe(recipe_path: str, append_path: str) -> None:
    with open(recipe_path) as f:
        recipe = f.readlines()

    with open(append_path) as f:
        append = f.readlines()

    with open(recipe_path, "w") as f:
        f.writelines(recipe + ["\n"] + append)


# # If want separate env for updating/reporting version with setuptools-scm
# # We do this from dev environment.
# # ** version report/update
# @DEFAULT_SESSION
# def version_scm(
#     session: Session,
#     version: VERSION_CLI = "",
#     update: UPDATE_CLI = False,
# ):
#     """
#     Get current version from setuptools-scm

#     Note that the version of editable installs can get stale.
#     This will show the actual current version.
#     Avoids need to include setuptools-scm in develop/docs/etc.
#     """

#     pkg_install_venv(
#         session=session,
#         name="version-scm",
#         install_package=True,
#         reqs=["setuptools_scm"],
#         update=update,
#         no_deps=True,
#     )

#     if version:
#         session.env["SETUPTOOLS_SCM_PRETEND_VERSION"] = version

#     session.run("python", "-m", "setuptools_scm")
