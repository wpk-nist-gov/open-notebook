"""
Light weight argparser from a dataclass.

There are some libraries out there that mostly do what we want, but none are quite right.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from dataclasses import (
    MISSING as dataclass_MISSING,
)
from dataclasses import (
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from typing import Any, Callable, Container, Literal, Sequence, Union, cast

assert sys.version_info >= (3, 10)

# if sys.version_info < (3, 10):  # pragma: no cover
#     from typing_extensions import Annotated, get_args, get_origin, get_type_hints
# else:
#     from typing import Annotated, get_args, get_origin, get_type_hints

from typing import Annotated, get_args, get_origin, get_type_hints

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self


_NoneType = type(None)

UNDEFINED = cast(
    Any, type("Undefined", (), {"__repr__": lambda self: "UNDEFINED"})()
)  # pyright: ignore


@dataclass
class Option:
    """Class to handle options."""

    flags: Sequence[str] = UNDEFINED
    # remainder are alphabetical
    action: str | None = UNDEFINED
    choices: Container[Any] = UNDEFINED
    const: Any = UNDEFINED
    default: Any | None = UNDEFINED
    dest: str = UNDEFINED
    help: str = UNDEFINED
    metavar: str = UNDEFINED
    nargs: str | int | None = UNDEFINED
    required: bool = UNDEFINED
    type: int | float | Callable[[Any], Any] = UNDEFINED
    prefix_char: str = "-"

    def __post_init__(self) -> None:
        if isinstance(self.flags, str):
            self.flags = (self.flags,)
        if self.flags is not UNDEFINED:
            for f in self.flags:
                if not f.startswith(self.prefix_char):
                    raise ValueError(f"Option only supports flags, but got {f!r}")

    def add_argument_to_parser(
        self, parser: ArgumentParser, prefix_char: str = "-"
    ) -> None:
        kwargs = {
            k: v
            for k, v in (
                # Can't use asdict() since that deep copies and we need
                # to filter using an identity check against UNDEFINED.
                [
                    (f.name, getattr(self, f.name))
                    for f in fields(self)
                    if f.name != "prefix_char"
                ]
            )
            if v is not UNDEFINED
        }

        flags = kwargs.pop("flags")

        # make sure flags have correct prefixing:
        if not all(flag.startswith(prefix_char) for flag in flags):
            new_flags: list[str] = []
            for flag in flags:
                if flag.startswith(prefix_char):
                    new_flags.append(flag)
                elif flag.startswith("--"):
                    new_flags.append(prefix_char * 2 + flag.lstrip("-"))
                elif flag.startswith("-"):
                    new_flags.append(prefix_char + flag.lstrip("-"))
                else:
                    raise ValueError(f"bad flag {flag} prefix_char {prefix_char}")

            flags = new_flags

        parser.add_argument(*flags, **kwargs)


def _is_union_type(t: Any) -> bool:
    # types.UnionType only exists in Python 3.10+.
    # https://docs.python.org/3/library/stdtypes.html#types-union
    if sys.version_info >= (3, 10):
        import types

        origin = get_origin(t)
        return origin is types.UnionType or origin is Union
    else:
        return False


def _get_underlying_if_optional(t: Any, pass_through: bool = False) -> Any:
    if _is_union_type(t):
        args = get_args(t)
        if len(args) == 2 and _NoneType in args:
            for t in args:
                if t != _NoneType:
                    return t
    elif pass_through:
        return t

    return None


def _get_choices(
    opt: Any, allow_optional: bool = True, allow_list: bool = True
) -> list[Any] | None:
    """Parse a type annotation for Literal[...], list[Literal[...]], or list[Literal[...]] | None"""
    out: list[Any] = []

    if (opt_origin := get_origin(opt)) is None:
        pass
    elif opt_origin is Literal:
        out.extend(get_args(opt))
    elif allow_list and opt_origin is list:
        opt_arg, *extras = get_args(opt)
        if not extras and (
            new_out := _get_choices(opt_arg, allow_optional=False, allow_list=False)
        ):
            out.extend(new_out)
    elif allow_optional and (underlying := _get_underlying_if_optional(opt)):
        if new_out := _get_choices(
            underlying, allow_optional=False, allow_list=allow_list
        ):
            out.extend(new_out)

    return out or None


def _get_underlying_type(
    opt: Any, allow_optional: bool = True, depth: int = 0, max_depth: int = 2
) -> tuple[int, Any]:
    """
    Parse nested list of type -> (depth, type)

    list[list[str]] -> (2, str)
    """
    depth_out = depth

    type_: Any = opt

    if depth > max_depth or ((opt_origin := get_origin(opt)) is None):
        pass

    elif opt_origin is list:
        opt_arg, *extras = get_args(opt)
        if not extras:
            depth_out, type_ = _get_underlying_type(
                opt_arg, allow_optional=False, depth=depth + 1, max_depth=max_depth
            )

    elif allow_optional and (underlying := _get_underlying_if_optional(opt)):
        depth_out, type_ = _get_underlying_type(
            underlying, allow_optional=False, depth=depth, max_depth=max_depth
        )

    return depth_out, type_


def _create_option(
    name: str, default: Any, annotation: Any, explicit_options: bool
) -> Option | None:
    if get_origin(annotation) is Annotated:
        anno_args = get_args(annotation)
    elif explicit_options:
        return None
    else:
        anno_args = (annotation, Option())

    opt: Option | None
    opt_type, opt, *extra_args = anno_args

    if extra_args:
        raise ValueError(f"{annotation} has extra metadata {extra_args}")

    if opt is None:
        return None

    # # Just just underlying type
    # if opt.choices is UNDEFINED:
    #     if (choices := _get_choices(opt_type)):
    #         opt = replace(opt, choices=choices)
    # else:
    #     choices = None

    depth, underlying_type = _get_underlying_type(opt_type)

    if depth <= 2 and get_origin(underlying_type) is Literal:
        choices = get_args(underlying_type)
        if opt.choices is UNDEFINED:
            opt = replace(opt, choices=choices)
    else:
        choices = None

    if opt.nargs is UNDEFINED and depth > 0:
        opt = replace(opt, nargs="*")

    if opt.action is UNDEFINED and depth > 1:
        opt = replace(opt, action="append")

    if opt.type is UNDEFINED:
        if choices:
            opt_type = type(choices[0])

        elif depth > 0:
            opt_type = underlying_type

        if not callable(opt_type):
            raise TypeError(
                f"Annotation {annotation} for parameter {name!r} is not callable."
                f"Declare option type with Annotated[..., Option(type=...)] instead."
            )
        opt = replace(opt, type=opt_type)

    if opt.type is bool:
        opt = replace(
            opt,
            action="store_false" if default is True else "store_true",
            type=UNDEFINED,
            default=UNDEFINED,
        )
    elif default is not UNDEFINED:
        opt = replace(opt, default=default)
    else:
        opt = replace(opt, required=True)

    # origin = get_origin(opt)
    # if origin is list:
    #     # if list type, then set nargs to "*"
    #     if opt.nargs is UNDEFINED:
    #         opt = replace(opt, nargs="*")

    #     # if list[Literal], set choices

    if opt.flags is UNDEFINED:
        opt = replace(opt, flags="--" + name.replace("_", "-"))
    return opt


def get_dataclass_options(cls: Any) -> dict[str, Option]:
    options: dict[str, Option] = {}

    for k, (d, a) in _get_dataclass_defaults_and_annotations(cls).items():
        opt = _create_option(k, d, a, False)
        if opt is not None:
            options[k] = opt

    return options


def _get_dataclass_defaults_and_annotations(cls: Any) -> dict[str, tuple[Any, Any]]:
    annotations = get_type_hints(cls, include_extras=True)

    assert is_dataclass(cls)
    cls_fields = fields(cls)

    out: dict[str, tuple[Any, Any]] = {}
    for f in cls_fields:
        if f.name.startswith("_") or not f.init:
            continue
        else:
            out[f.name] = (
                UNDEFINED if f.default is dataclass_MISSING else f.default,
                annotations[f.name],
            )
    return out


class DataClassParserMixin:
    """Mixin for creating a dataclass with parsing."""

    @classmethod
    def parser(cls, prefix_char: str = "-", **kwargs: Any) -> ArgumentParser:
        parser = ArgumentParser(prefix_chars=prefix_char, **kwargs)

        for _, opt in get_dataclass_options(cls).items():
            opt.add_argument_to_parser(parser, prefix_char=prefix_char)

        return parser

    @classmethod
    def from_posargs(
        cls,
        posargs: str | Sequence[str],
        prefix_char: str = "-",
        parser: ArgumentParser | None = None,
        known: bool = False,
    ) -> Self:
        if parser is None:
            parser = cls.parser(prefix_char=prefix_char)

        if isinstance(posargs, str):
            import shlex

            posargs = shlex.split(posargs)

        if known:
            parsed, _ = parser.parse_known_args(posargs)
        else:
            parsed = parser.parse_args(posargs)

        return cls(**vars(parsed))
