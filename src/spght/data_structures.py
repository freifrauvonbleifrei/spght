# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Prototype API for an adaptive hierarchical sparse grid file format.
The binary encoding details may still evolve.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Sequence

from spght.lifting import Basis1D
from spght.linearize import Order
from spght.tensor import Tensor, TensorKind
from spght.util import per_dimension
from spght.wavelets import haar_basis


@dataclass(frozen=True, slots=True)
class Subspace:
    """Describe one logical subspace in memory and on disk."""

    extents: tuple[int, ...]
    # informational in v0.1: values are stored at their dtype's width,
    # a custom precision is not (yet) enforced
    precision_bits: int
    data: Tensor | None = None
    # three quantization parameters, reserved for future use
    quantization_scale: float = 1.0
    quantization_offset: float = 0.0
    quantization_parameter: float = 0.0
    padding_bits: int = 0
    compression: int = 0

    @property
    def order(self) -> Order:
        return self.data.order if self.data is not None else "C"

    @property
    def kind(self) -> TensorKind:
        """Storage kind of the subspace data (EMPTY when no data is attached)."""
        return self.data.kind if self.data is not None else TensorKind.EMPTY

    @property
    def is_sparse(self) -> bool:
        return self.data.is_sparse if self.data is not None else False

    @property
    def num_bytes(self) -> int:
        return self.data.nbytes if self.data is not None else 0

    def __post_init__(self) -> None:
        if self.precision_bits <= 0:
            raise ValueError("precision_bits must be positive")
        if self.padding_bits < 0:
            raise ValueError("padding_bits must not be negative")
        if self.data is not None and tuple(self.data.shape) != tuple(self.extents):
            raise ValueError(
                f"data shape {self.data.shape} does not match extents {self.extents}"
            )


def subspace_order_key(level: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """Canonical subspace ordering: ascending level sum (coarse to fine),
    ties broken lexicographically (dimension 0 most significant)."""
    level_tuple = tuple(level)
    return (sum(level_tuple), level_tuple)


@dataclass(slots=True)
class SparseGridHierarchicalTensors:
    dimensions: int
    max_level: tuple[int, ...]
    # minimum level per dimension: where the wavelet cascade stops; along
    # dimension d, subspaces with level[d] == min_level[d] hold scaling
    # coefficients.
    min_level: tuple[int, ...] = ()
    bases: tuple[Basis1D, ...] = ()
    subspaces: dict[tuple[int, ...], Subspace] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.max_level) != self.dimensions:
            raise ValueError("max_level dimensionality does not match container")
        self.min_level = per_dimension(
            self.min_level if self.min_level != () else 0,
            int,
            self.dimensions,
            "min_level",
        )
        self.bases = per_dimension(
            self.bases if self.bases != () else haar_basis(),
            Basis1D,
            self.dimensions,
            "basis",
        )
        if any(
            minimum > maximum
            for minimum, maximum in zip(self.min_level, self.max_level)
        ):
            raise ValueError(
                f"min_level {self.min_level} exceeds max_level {self.max_level}"
            )
        for k, v in self.subspaces.items():
            if len(k) != self.dimensions:
                raise ValueError("subspace key dimensionality does not match container")
            if any(level < minimum for level, minimum in zip(k, self.min_level)):
                raise ValueError(f"subspace level {k} is below min_level")
            if len(v.extents) != self.dimensions:
                raise ValueError(
                    "subspace extents dimensionality does not match container"
                )
        self._sort_subspaces()

    def scaling_dimensions(self, level: Sequence[int]) -> tuple[bool, ...]:
        """Along which dimensions a subspace at `level` holds scaling
        (approximation) coefficients: exactly those where the level equals
        min_level; all finer levels hold detail coefficients."""
        return tuple(
            single_level == minimum
            for single_level, minimum in zip(level, self.min_level)
        )

    def _sort_subspaces(self) -> None:
        """Restore the canonical (level sum, lexicographic) subspace order."""
        self.subspaces = dict(
            sorted(self.subspaces.items(), key=lambda item: subspace_order_key(item[0]))
        )

    def add_subspace(self, level: tuple[int, ...], subspace: Subspace) -> None:
        if len(level) != self.dimensions:
            raise ValueError("subspace dimensionality does not match container")
        if len(self.max_level) != self.dimensions:
            raise ValueError("max_level dimensionality does not match container")
        if any(
            single_level < minimum
            for single_level, minimum in zip(level, self.min_level)
        ):
            raise ValueError(f"subspace level {level} is below min_level")
        self.subspaces[level] = subspace
        self._sort_subspaces()

    def write(self, target: "str | Path | BinaryIO") -> None:
        """Write the hierarchy to disk in the v0.3 binary layout
        (see spght.serialize for the format description)."""
        # imported lazily: serialize imports this class
        from spght.serialize import write

        write(self, target)

    @classmethod
    def read(cls, source: "str | Path | BinaryIO") -> "SparseGridHierarchicalTensors":
        """Load a hierarchy from disk."""
        from spght.serialize import read

        return read(source)
