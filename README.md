<!-- markdownlint-disable MD041 -->

[![Repo][repo-badge]][repo-link] [![Docs][docs-badge]][docs-link]
[![PyPI license][license-badge]][license-link]
[![PyPI version][pypi-badge]][pypi-link]
[![Conda (channel only)][conda-badge]][conda-link]
[![Code style: black][black-badge]][black-link]

<!--
  For more badges, see
  https://shields.io/category/other
  https://naereen.github.io/badges/
  [pypi-badge]: https://badge.fury.io/py/open-notebook
-->

<!-- prettier-ignore-start -->
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[black-link]: https://github.com/psf/black
[pypi-badge]: https://img.shields.io/pypi/v/open-notebook
[pypi-link]: https://pypi.org/project/open-notebook
[docs-badge]: https://img.shields.io/badge/docs-sphinx-informational
[docs-link]: https://pages.nist.gov/open-notebook/
[repo-badge]: https://img.shields.io/badge/--181717?logo=github&logoColor=ffffff
[repo-link]: https://github.com/usnistgov/open-notebook
[conda-badge]: https://img.shields.io/conda/v/wpk-nist/open-notebook
[conda-link]: https://anaconda.org/wpk-nist/open-notebook
[license-badge]: https://img.shields.io/pypi/l/cmomy?color=informational
[license-link]: https://github.com/usnistgov/open-notebook/blob/main/LICENSE
<!-- prettier-ignore-end -->

<!-- other links -->

[jupyter]: https://jupyter.org/

# `open-notebook`

A python cli program to open [jupyter] notebooks from a central server.

## Overview

I typically run a single [jupyter] notebook server, and want to launch notebooks
from several locations in the file system. This can be done from the `tree` view
of jupyter notebook, but I want to quickly open notebooks from the command line
using the central server. `open-notebook` will open notebooks from anywhere
relative to a central server.

## Features

- Can specify options like `host`, and `port` from the command line
- Defaults can be configured using the configuration file(s)
  `.open-notebook.toml`.
- Open both [jupyter] notebooks, and directories (in tree view).

## Status

This package is actively used by the author. Please feel free to create a pull
request for wanted features and suggestions!

## Quick start

Use one of the following

```bash
pip install open-notebook
```

or

```bash
conda install -c wpk-nist open-notebook
```

## Example usage

<!-- markdownlint-disable-next-line MD013 -->
<!-- [[[cog
import sys
sys.path.insert(0, ".")
from tools.cog_utils import wrap_command, get_pyproject, run_command, cat_lines
sys.path.pop(0)
]]] -->
<!-- [[[end]]] -->

### Options

The main command-line program is `open-notebook` with the following options:

<!-- prettier-ignore-start -->
<!-- markdownlint-disable MD013 -->
<!-- [[[cog run_command("open-notebook --help", include_cmd=True, wrapper="bash")]]] -->
```bash
$ open-notebook --help
usage: open-notebook [-h] [--host HOST] [-p PORT] [-r ROOT] [--dir-prefix DIR_PREFIX]
                     [--file-prefix FILE_PREFIX] [--reset] [-c CONFIG] [--create-config]
                     [--overwrite] [--version] [-v] [--dry]
                     [paths ...]

Program to open jupyter notebooks from central notebook server.

positional arguments:
  paths                 file or paths to open

options:
  -h, --help            show this help message and exit
  --host HOST           Host name (default='localhost')
  -p PORT, --port PORT  Port (default='8888')
  -r ROOT, --root ROOT  Directory servers was started in. Defaults to current working
                        directory.
  --dir-prefix DIR_PREFIX
                        Directory prefix (default='tree')
  --file-prefix FILE_PREFIX
                        File prefix (default='notebooks')
  --reset
  -c CONFIG, --config CONFIG
                        Config style to use. This is the name of a header in one of the
                        config files.
  --create-config       If passed, create .open-notebook.ini
  --overwrite           Pass to overwrite `~/.open-notebook.toml` if it exists with
                        `--create-config`
  --version             Print out program version
  -v, --verbose         Set verbosity level. Can pass multiple times.
  --dry                 Dry run.

You can set options with the configuration files ".open-notebook.toml". Configuration
files are found in the current directory, git root (if in a git tracked tree), and the
home directory. Note that all these files are considered, in order. That is, you could
override a single value in the current directory, and the rest would be inherited from,
in order, git root and then the home directory.
```

<!-- [[[end]]] -->
<!-- prettier-ignore-end -->

<!-- markdownlint-enable MD013 -->

Equivalently, you can use the short name `nopen`, or use
`python -m open_notebook`.

### Basic usage

To open directory tree view:

```bash
open-notebook .
```

To open a jupyter notebook (here using the short name `nopen`):

```bash
nopen path/to/notebook.ipynb
```

To specify where the central server is running, use the `-r/--root` option. For
example, if the server is started in the directory "~/test", then you'd pass
`--root ~/test`. For example:

```bash
# start server
cd ~/test
jupyter notebook


# cd to some other directory under where notebook was started
cd ~/test/a/different/directory
open-notebook -r ~/test example.ipynb
```

### Configuration file

If you always start a notebook in the same place, you can configure
`open-notebook` as follows:

```toml
# ~/.open-notebook.toml
root = "~/test"
port = "8888"

```

Options name in the configuration file `.open-notebook.toml` are the same as the
command-line options above (replacing dashes with underscores, so, e.g., instead
of `--dir-prefix value`, you'd sed `dir_prefix = "value"` in the configuration
file).

If you use more than one server, you can have multiple notebook configurations.
For example, you can specify that the configuration `alt` uses port `8889` and
is run in the home directory using:

```toml
# ~/.open-notebook.toml
root = "~/test"
port = "8888"

[alt]
root = "~/"
port = "8889"

```

The default behavior is the same as above. To use the `alt` config, then use:

```bash
# will use root="~/" and port="8889".  Other options inherited.
open-notebook -c alt ...
```

### Multiple configuration files

`open-notebook` searches for configuration files `.open-notebook.toml` in the
current directory, the root of a git repo (if you're currently in a git repo),
and finally in the home directory. Options are read, in order, from command-line
options, current directory config, git root config, and home directory config
file. This means that you can specify common configurations at the home level,
and then override single options at higher levels. For example, if we have:

```toml
# ~/.open-notebook.toml
root = "~/"
host = "8889"

```

```toml
# ~/a/b/.open-notebook.toml
host = "9999"

```

```bash
cd a/b
# this will open notebook with root="~/" and host="9999"
open-notebook example.ipynb
```

<!-- end-docs -->

## Documentation

See the [documentation][docs-link] for further details.

## License

This is free software. See [LICENSE][license-link].

## Contact

The author can be reached at <wpk@nist.gov>.

## Credits

This package was created using
[Cookiecutter](https://github.com/audreyr/cookiecutter) with the
[usnistgov/cookiecutter-nist-python](https://github.com/usnistgov/cookiecutter-nist-python)
template.
