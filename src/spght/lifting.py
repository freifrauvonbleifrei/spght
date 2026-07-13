# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Lifting-scheme wavelet transforms over centerings and boundary rules."""

import itertools
from dataclasses import dataclass, field
from typing import Literal, Sequence, Union

import numpy as np
import numpy.typing as npt

from spght.basis import (
    BoundaryRule,
    CellCentered,
    Centering,
    Dirichlet,
    Extrapolate,
    Periodic,
    VertexCentered,
    WallType,
)


@dataclass(frozen=True)
class LiftingStep:
    kind: Literal["predict", "update", "scale_detail"]
    offsets: tuple[int, ...] = ()
    weights: tuple[float, ...] = ()
    factor: float = 1.0


@dataclass(frozen=True)
class LiftingScheme:
    """The wavelet itself, written as predict/update/scale steps."""

    name: str
    steps: tuple[LiftingStep, ...]
    centering_kind: Literal["cell", "vertex"]
    evaluation: Literal["midpoint", "nodal_linear"] | None


@dataclass(frozen=True)
class Basis1D:
    """Everything one dimension of a hierarchical transform needs."""

    centering: Centering
    scheme: LiftingScheme
    bc_left: BoundaryRule = field(default_factory=Extrapolate)
    bc_right: BoundaryRule = field(default_factory=Extrapolate)

    def __post_init__(self) -> None:
        is_cell = isinstance(self.centering, CellCentered)
        if self.scheme.centering_kind != ("cell" if is_cell else "vertex"):
            raise ValueError(
                f"scheme {self.scheme.name} is {self.scheme.centering_kind}-"
                f"centered but the centering is {self.centering.name}"
            )
        interior_only = (
            isinstance(self.centering, VertexCentered)
            and not self.centering.include_boundary
        )
        if interior_only and not all(
            isinstance(bc, Dirichlet) for bc in (self.bc_left, self.bc_right)
        ):
            raise ValueError(
                "interior-only vertex grids need boundary rules that determine "
                "the wall values (Dirichlet)"
            )
        if getattr(self.centering, "periodic", False) and not all(
            isinstance(bc, Periodic) for bc in (self.bc_left, self.bc_right)
        ):
            raise ValueError(
                "a periodic vertex grid requires Periodic rules on both sides"
            )
        if isinstance(self.bc_left, Periodic) != isinstance(self.bc_right, Periodic):
            raise ValueError("periodic boundaries must be used on both sides")

    def homogenized(self) -> "Basis1D":
        return Basis1D(
            self.centering,
            self.scheme,
            self.bc_left.homogenized(),
            self.bc_right.homogenized(),
        )


BasisLike = Union[Basis1D, Sequence[Basis1D]]


def as_bases(basis: BasisLike, num_dim: int) -> tuple[Basis1D, ...]:
    """Normalize a single basis or a per-dimension sequence to a tuple."""
    if isinstance(basis, Basis1D):
        return (basis,) * num_dim
    bases = tuple(basis)
    if len(bases) != num_dim or not all(isinstance(b, Basis1D) for b in bases):
        raise ValueError(
            f"expected one Basis1D or a sequence of {num_dim}, got {basis!r}"
        )
    return bases


def _padded(
    arr: npt.NDArray,
    axis: int,
    pad_left: int,
    pad_right: int,
    bc_left: BoundaryRule,
    bc_right: BoundaryRule,
    wall: WallType,
) -> npt.NDArray:
    if pad_left == 0 and pad_right == 0:
        return arr
    extent = arr.shape[axis]
    slabs = (
        [bc_left.ghost_slab(arr, pos, wall, axis) for pos in range(-pad_left, 0)]
        + [arr]
        + [bc_right.ghost_slab(arr, extent + k, wall, axis) for k in range(pad_right)]
    )
    return np.concatenate(slabs, axis=axis)


def _stencil_sum(
    source: npt.NDArray,
    axis: int,
    target_extent: int,
    shift: int,
    step: LiftingStep,
    bc_left: BoundaryRule,
    bc_right: BoundaryRule,
    wall: WallType,
) -> npt.NDArray:
    """sum_k weights[k] * source[j + shift + offsets[k]] for each target j,
    with out-of-range taps answered by the boundary rules."""
    extent = source.shape[axis]
    pad_left = max(0, -(shift + min(step.offsets)))
    pad_right = max(0, target_extent - 1 + shift + max(step.offsets) - (extent - 1))
    padded = _padded(source, axis, pad_left, pad_right, bc_left, bc_right, wall)
    total: npt.NDArray | None = None
    for offset, weight in zip(step.offsets, step.weights):
        start = shift + offset + pad_left
        taps = [slice(None)] * source.ndim
        taps[axis] = slice(start, start + target_extent)
        term = weight * padded[tuple(taps)]
        total = term if total is None else total + term
    assert total is not None
    return total


def _split(values: npt.NDArray, axis: int, centering: Centering):
    first = [slice(None)] * values.ndim
    second = [slice(None)] * values.ndim
    first[axis] = slice(0, None, 2)
    second[axis] = slice(1, None, 2)
    even, odd = values[tuple(first)], values[tuple(second)]
    return (even, odd) if centering.even_first else (odd, even)


def _merge(
    even: npt.NDArray, odd: npt.NDArray, axis: int, centering: Centering
) -> npt.NDArray:
    shape = list(even.shape)
    shape[axis] = even.shape[axis] + odd.shape[axis]
    merged = np.empty(shape, dtype=np.result_type(even, odd))
    first = [slice(None)] * merged.ndim
    second = [slice(None)] * merged.ndim
    first[axis] = slice(0, None, 2)
    second[axis] = slice(1, None, 2)
    if centering.even_first:
        merged[tuple(first)], merged[tuple(second)] = even, odd
    else:
        merged[tuple(first)], merged[tuple(second)] = odd, even
    return merged


def lifting_step(
    even: npt.NDArray,
    odd: npt.NDArray,
    axis: int,
    basis: Basis1D,
    inverse: bool = False,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Apply one level of the scheme's lifting steps along `axis`."""
    even, odd = np.array(even, dtype=float), np.array(odd, dtype=float)
    centering = basis.centering
    steps = reversed(basis.scheme.steps) if inverse else basis.scheme.steps
    sign = -1.0 if inverse else 1.0
    for step in steps:
        if step.kind == "predict":
            odd -= sign * _stencil_sum(
                even,
                axis,
                odd.shape[axis],
                centering.even_shift,
                step,
                basis.bc_left,
                basis.bc_right,
                centering.even_wall,
            )
        elif step.kind == "update":
            # details carry no boundary data: use the homogenized rules
            even += sign * _stencil_sum(
                odd,
                axis,
                even.shape[axis],
                centering.odd_shift,
                step,
                basis.bc_left.homogenized(),
                basis.bc_right.homogenized(),
                centering.odd_wall,
            )
        else:  # scale_detail
            odd *= step.factor if not inverse else 1.0 / step.factor
    return even, odd


def decompose_axis(
    values: npt.NDArray, axis: int, basis: Basis1D, min_level: int
) -> list[npt.NDArray]:
    """Multilevel decomposition along `axis`. Returns the blocks coarsest
    first: [scaling at min_level, details at min_level + 1, ..., details at
    the maximum level] (the same order pywt.wavedec uses)."""
    level = basis.centering.level_from_num_dofs(values.shape[axis])
    if min_level < basis.centering.lowest_min_level:
        raise ValueError(
            f"min_level {min_level} is below the coarsest level "
            f"{basis.centering.lowest_min_level} of {basis.centering.name}"
        )
    details = []
    current = np.asarray(values, dtype=float)
    for _ in range(level, min_level, -1):
        even, odd = _split(current, axis, basis.centering)
        even, odd = lifting_step(even, odd, axis, basis)
        details.append(odd)
        current = even
    return [current] + details[::-1]


def reconstruct_axis(
    blocks: Sequence[npt.NDArray], axis: int, basis: Basis1D
) -> npt.NDArray:
    """Inverse of decompose_axis; `blocks` ordered coarsest first."""
    current = np.asarray(blocks[0], dtype=float)
    for odd in blocks[1:]:
        even, odd = lifting_step(current, np.asarray(odd), axis, basis, inverse=True)
        current = _merge(even, odd, axis, basis.centering)
    return current


def reconstruct_detail_block(
    coefficients: npt.NDArray, axis: int, basis: Basis1D
) -> npt.NDArray:
    """A single detail block's contribution as nodal values on its own
    level's grid: one inverse lifting step with zero scaling input and
    homogenized boundary rules (boundary data belongs to the scaling
    subspaces)."""
    centering = basis.centering
    level = centering.level_from_num_details(coefficients.shape[axis])
    even_shape = list(coefficients.shape)
    even_shape[axis] = centering.num_dofs(level - 1)
    even = np.zeros(even_shape)
    even, odd = lifting_step(
        even, coefficients, axis, basis.homogenized(), inverse=True
    )
    return _merge(even, odd, axis, centering)


def evaluation_taps(
    coordinates_1d: npt.NDArray,
    extent: int,
    basis: Basis1D,
) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """Per-coordinate (index, weight) taps into a nodal block of `extent`
    dofs along one dimension. Interior-only blocks must be padded with
    `wall_values` by the caller first, so taps never leave the block."""
    if basis.scheme.evaluation == "midpoint":
        indices = np.clip((coordinates_1d * extent).astype(np.int64), 0, extent - 1)
        return [(indices, np.ones_like(coordinates_1d))]
    if basis.scheme.evaluation == "nodal_linear":
        if getattr(basis.centering, "periodic", False):
            position = coordinates_1d * extent
            lower = np.floor(position).astype(np.int64)
            fraction = position - lower
            lower %= extent
            return [(lower, 1.0 - fraction), ((lower + 1) % extent, fraction)]
        position = coordinates_1d * (extent - 1)
        lower = np.clip(np.floor(position).astype(np.int64), 0, extent - 2)
        fraction = position - lower
        return [(lower, 1.0 - fraction), (lower + 1, fraction)]
    raise NotImplementedError(
        f"scheme {basis.scheme.name} does not support direct evaluation"
    )


def wall_values(basis: Basis1D, scaling: bool) -> tuple[float, float] | None:
    """For interior-only vertex grids: the wall values to pad an evaluation
    block with (None when the block already covers the domain)."""
    centering = basis.centering
    if not (isinstance(centering, VertexCentered) and not centering.include_boundary):
        return None
    if not scaling:
        return (0.0, 0.0)
    assert isinstance(basis.bc_left, Dirichlet)
    assert isinstance(basis.bc_right, Dirichlet)
    return (basis.bc_left.value, basis.bc_right.value)


def evaluate_block(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    coefficients: npt.NDArray,
    bases: Sequence[Basis1D],
) -> npt.NDArray:
    """Evaluate one subspace's contribution at `coordinates` (shape (n, d)):
    reconstruct each detail dimension one level, then take the tensor
    product of the per-dimension evaluation stencils."""
    block = np.asarray(coefficients, dtype=float)
    for d, (scaling, basis) in enumerate(zip(scaling_dimensions, bases)):
        if basis.scheme.evaluation is None:
            raise NotImplementedError(
                f"scheme {basis.scheme.name} does not support direct evaluation"
            )
        if not scaling:
            block = reconstruct_detail_block(block, d, basis)
        walls = wall_values(basis, scaling)
        if walls is not None:
            shape = list(block.shape)
            shape[d] = 1
            block = np.concatenate(
                [np.full(shape, walls[0]), block, np.full(shape, walls[1])], axis=d
            )
    taps = [
        evaluation_taps(coordinates[:, d], block.shape[d], basis)
        for d, basis in enumerate(bases)
    ]
    values = np.zeros(coordinates.shape[0])
    for combination in itertools.product(*taps):
        weights = np.ones(coordinates.shape[0])
        for _, weight in combination:
            weights = weights * weight
        values += weights * block[tuple(index for index, _ in combination)]
    return values
