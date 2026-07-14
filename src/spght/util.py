# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared across spght modules."""

from functools import wraps
from importlib.util import find_spec
from typing import Callable, Sequence, TypeVar, Union

T = TypeVar("T")


def module_is_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def depends_on_optional(module_name: str) -> Callable:
    """Mark a function as requiring an optional dependency: calling it
    without the module installed raises a helpful ImportError instead of
    failing at import time."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not module_is_available(module_name):
                raise ImportError(
                    f"Optional dependency {module_name} not found ({func.__name__})."
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator


def per_dimension(
    value: Union[T, Sequence[T]],
    scalar_type: Union[type, tuple[type, ...]],
    num_dim: int,
    what: str = "value",
) -> tuple[T, ...]:
    """Normalize a scalar-or-per-dimension argument to a tuple with one
    entry per dimension: a scalar is repeated `num_dim` times, a sequence
    is validated to hold exactly `num_dim` entries of the scalar type."""
    if isinstance(value, scalar_type):
        return (value,) * num_dim  # type: ignore[return-value]
    values = tuple(value)  # type: ignore[arg-type]
    if len(values) != num_dim or not all(
        isinstance(entry, scalar_type) for entry in values
    ):
        raise ValueError(
            f"expected one {what} or a sequence of {num_dim}, got {value!r}"
        )
    return values
