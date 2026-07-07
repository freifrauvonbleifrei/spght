"""Prototype API for an adaptive hierarchical sparse grid file format.
The binary encoding details may still evolve.
"""

from dataclasses import dataclass, field
import numpy.typing as npt
from pathlib import Path
from typing import BinaryIO, Literal, Sequence, cast


FormatMagic = b"sparse grid hierarchical tensors\0"
FormatVersion: tuple[int, int] = (0, 1)


@dataclass(frozen=True, slots=True)
class Subspace:
    """Describe one logical subspace in memory and on disk."""

    extents: tuple[int, ...]
    precision_bits: int
    # values: bytes
    values: npt.NDArray | None = None
    order: Literal["C", "F", "ZC", "ZF"] = "ZC"
    index_kind: str = "full"
    padding_bits: int = 0
    indices: Sequence[int] | Sequence[tuple[int, int]] | None = None
    checksum: int = 0
    compression: int = 0

    @property
    def num_bytes(self) -> int:
        if self.values is None:
            return 0
        return len(self.values)

    def __post_init__(self) -> None:
        if self.precision_bits <= 0:
            raise ValueError("precision_bits must be positive")
        if self.padding_bits < 0:
            raise ValueError("padding_bits must not be negative")


@dataclass(slots=True)
class SparseGridHierarchicalTensors:
    magic = FormatMagic
    dimensions: int
    max_level: tuple[int, ...]
    # levels: tuple[int, ...]
    subspaces: dict[tuple[int, ...], Subspace] = field(default_factory=dict)
    version = FormatVersion

    def __post_init__(self) -> None:
        for k, v in self.subspaces.items():
            if len(k) != self.dimensions:
                raise ValueError("subspace key dimensionality does not match container")
            if len(v.extents) != self.dimensions:
                raise ValueError(
                    "subspace extents dimensionality does not match container"
                )

    def add_subspace(self, level: tuple[int, ...], subspace: Subspace) -> None:
        if len(level) != self.dimensions:
            raise ValueError("subspace dimensionality does not match container")
        if len(self.max_level) != self.dimensions:
            raise ValueError("max_level dimensionality does not match container")
        self.subspaces[level] = subspace


def open_file(path: str | Path, mode: str = "rb") -> BinaryIO:
    """Small wrapper for file access used by the prototype API."""

    return cast(BinaryIO, Path(path).open(mode))
