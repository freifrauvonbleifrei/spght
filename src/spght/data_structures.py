# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Prototype API for an adaptive hierarchical sparse grid file format.
The binary encoding details may still evolve.
"""

from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO, Sequence, Union

import numpy as np

from spght.lifting import Basis1D
from spght.linearize import Order
from spght.tensor import Tensor, TensorKind
from spght.util import per_dimension
from spght.wavelets import haar_basis


@dataclass(frozen=True, slots=True)
class OpaqueValue:
    """A metadata entry whose value tag this version of spght does not
    know. Don't touch."""

    tag: int
    payload: bytes


# metadata values as they come back from a file: numeric sequences are
# normalized to tuples so that containers compare cleanly
MetadataValue = Union[
    str,
    float,
    int,
    bytes,
    Sequence[str],
    Sequence[float],
    Sequence[int],
    OpaqueValue,
]


class Convention(IntEnum):
    """Special-value convention of a number format; implies the exponent
    bias for floats."""

    IEEE = 0  # bias 2^(e-1) - 1; infinities and NaNs at all-ones exponent
    FN = 1  # finite only: same bias, no infinities, a single NaN (E4M3FN)
    FNUZ = 2  # finite, bias 2^(e-1), NaN at the negative-zero pattern
    OTHER = 3  # not parametric (e.g. posits); grid unknown, values exact


@dataclass(frozen=True, slots=True)
class NumberFormat:
    """Flavor of a subspace's stored number format, one byte on disk."""

    exponent_bits: int = 0
    unsigned: bool = False
    convention: Convention = Convention.IEEE

    def __post_init__(self) -> None:
        if not 0 <= self.exponent_bits <= 31:
            raise ValueError(
                f"exponent_bits must fit in 5 bits, got {self.exponent_bits}"
            )
        convention = Convention(self.convention)  # raises on unknown values
        if convention in (Convention.FN, Convention.FNUZ) and self.exponent_bits == 0:
            raise ValueError(
                "The FN and FNUZ conventions describe floats and need "
                "exponent_bits >= 1"
            )
        if convention == Convention.OTHER and (self.exponent_bits or self.unsigned):
            raise ValueError(
                "The OTHER convention is stored canonically: "
                "exponent_bits == 0 and signed"
            )

    def to_byte(self) -> int:
        """Pack into the on-disk byte: bits 0-4 exponent width, bit 5
        unsigned flag, bits 6-7 convention."""
        return self.exponent_bits | int(self.unsigned) << 5 | int(self.convention) << 6

    @classmethod
    def from_byte(cls, byte: int) -> "NumberFormat":
        return cls(byte & 0x1F, bool(byte >> 5 & 1), Convention(byte >> 6 & 0x3))

    @classmethod
    def from_dtype(cls, dtype: "np.dtype | type") -> "NumberFormat":
        """The native flavor of a numpy dtype (e.g. float32 -> 8 exponent
        bits, IEEE)."""
        dtype = np.dtype(dtype)
        if dtype.kind == "f":
            return cls(exponent_bits=np.finfo(dtype).nexp)
        if dtype.kind == "i":
            return cls()
        if dtype.kind == "u":
            return cls(unsigned=True)
        return cls(convention=Convention.OTHER)


@dataclass(frozen=True, slots=True)
class Subspace:
    """Describe one logical subspace in memory and on disk."""

    extents: tuple[int, ...]
    # the exact number of bits each stored value occupies in a file's
    # packed value stream (the data blob after any compression is
    # undone), 1..255. Until the bit-packing codec is implemented this
    # must equal the container dtype's width (8 * itemsize).
    precision_bits: int
    data: Tensor | None = None
    # the flavor of the (precision_bits)-wide format the values live on;
    # derived from the data dtype (or float64 without data) when None
    number_format: NumberFormat | None = None
    # reserved for future use (multiwavelets, per-subspace p-adaptivity):
    # number of values stored per spatial point
    num_components: int = 1
    # how a multi-component blob is arranged:
    # 0 = component-major planes (SoA), 1 = interleaved per point (AoS)
    component_layout: int = 0
    # two quantization parameters, reserved for future use with the
    # semantics logical = scale * (stored - zero_point)
    quantization_scale: float = 1.0
    quantization_zero_point: int = 0
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
        if not 1 <= self.precision_bits <= 255:
            raise ValueError(
                f"precision_bits must fit in one byte, got {self.precision_bits}"
            )
        if self.number_format is None:
            derived = (
                NumberFormat.from_dtype(self.data.dtype)
                if self.data is not None
                else NumberFormat.from_dtype(np.float64)
            )
            object.__setattr__(self, "number_format", derived)
        number_format = self.number_format
        assert number_format is not None
        sign_bit = 0 if number_format.unsigned else 1
        if sign_bit + number_format.exponent_bits > self.precision_bits:
            raise ValueError(
                f"number format {number_format} needs at least "
                f"{sign_bit + number_format.exponent_bits} bits, but "
                f"precision_bits is {self.precision_bits}"
            )
        if self.padding_bits < 0:
            raise ValueError("padding_bits must not be negative")
        if not 1 <= self.num_components <= 255:
            raise ValueError(
                f"num_components must fit in one byte, got {self.num_components}"
            )
        if self.component_layout not in (0, 1):
            raise ValueError(
                "component_layout must be 0 (planes) or 1 (interleaved), "
                f"got {self.component_layout}"
            )
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
    # descriptive key-value metadata (field name, domain bounds, ...);
    # carried and serialized, but not interpreted by spght, see serialize.py
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

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
            self._validate_subspace(k, v)
        self._sort_subspaces()

    def _validate_subspace(self, level: tuple[int, ...], subspace: Subspace) -> None:
        if len(level) != self.dimensions:
            raise ValueError("subspace key dimensionality does not match container")
        if any(
            single_level < minimum
            for single_level, minimum in zip(level, self.min_level)
        ):
            raise ValueError(f"subspace level {level} is below min_level")
        if len(subspace.extents) != self.dimensions:
            raise ValueError("subspace extents dimensionality does not match container")

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
        self._validate_subspace(level, subspace)
        self.subspaces[level] = subspace
        self._sort_subspaces()

    def relinearize(self, order: Order) -> "SparseGridHierarchicalTensors":
        """Re-linearize every subspace's buffer into `order`, in place. Returns self for chaining."""
        for level, subspace in self.subspaces.items():
            if subspace.data is not None and subspace.data.order != order:
                self.subspaces[level] = replace(
                    subspace, data=subspace.data.with_order(order)
                )
        return self

    def write(self, target: "str | Path | BinaryIO") -> None:
        """Write the hierarchy to disk in the spght binary format
        (see spght.serialize for the layout description)."""
        # imported lazily: serialize imports this class
        from spght.serialize import write

        write(self, target)

    @classmethod
    def read(cls, source: "str | Path | BinaryIO") -> "SparseGridHierarchicalTensors":
        """Load a hierarchy from disk."""
        from spght.serialize import read

        return read(source)
