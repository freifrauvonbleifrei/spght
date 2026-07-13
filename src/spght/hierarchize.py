# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

# Hierarchize a multi-dimensional function on a structured grid using the unidirectional principle.

import itertools
import numpy as np
import numpy.typing as npt
from typing import Sequence

import spght.data_structures as data_structures
from spght.lifting import (
    Basis1D,
    BasisLike,
    as_bases,
    decompose_axis,
    reconstruct_axis,
)
from spght.tensor import DenseTensor
from spght.wavelets import haar_basis


def _normalized_min_level(min_level: int | Sequence[int], num_dim: int) -> list[int]:
    if isinstance(min_level, int):
        return [min_level] * num_dim
    return list(min_level)


def hierarchize(
    nodal_values: npt.NDArray,
    wavelet: BasisLike | None = None,
    min_level: int | Sequence[int] = 0,
) -> data_structures.SparseGridHierarchicalTensors:
    """Decompose nodal values into hierarchical subspaces.

    `wavelet` is a Basis1D (or one per dimension); the default is the
    cell-centered Haar basis. The basis is recorded on the returned
    container, so reconstruction does not need it passed again."""
    num_dim = nodal_values.ndim
    minimum_levels = _normalized_min_level(min_level, num_dim)
    level: npt.NDArray = np.ndarray(num_dim, dtype=int)

    if wavelet is None:
        wavelet = haar_basis()
    bases = as_bases(wavelet, num_dim)
    for d in range(num_dim):
        level[d] = bases[d].centering.level_from_num_dofs(nodal_values.shape[d])
        lowest = bases[d].centering.lowest_min_level
        if not lowest <= minimum_levels[d] <= level[d]:
            raise ValueError(
                f"min_level {minimum_levels} must be between {lowest} and "
                f"the maximum level {tuple(level[: d + 1])} in every dimension"
            )

    modified_values = [nodal_values]
    for d in range(num_dim):
        updated_values = []
        for slices in modified_values:
            updated_values.extend(decompose_axis(slices, d, bases[d], minimum_levels[d]))
        modified_values = updated_values

    # construct a matching list of subspace levels: tensor product of 1D levels
    # from min_level[d] to level[d] for each dimension d. The order must match
    # the order in which `modified_values` was built by the nested transform
    # loop above, i.e. C-order (the coarsest block comes first).
    subspace_levels = list(
        itertools.product(
            *(range(minimum_levels[d], level[d] + 1) for d in range(num_dim))
        )
    )

    return data_structures.SparseGridHierarchicalTensors(
        dimensions=num_dim,
        max_level=tuple(level),
        min_level=tuple(minimum_levels),
        bases=bases,
        subspaces={
            # construction from a full array always yields dense (linear)
            # storage; sparsification only happens in compress()
            tuple(lv): data_structures.Subspace(
                extents=v.shape,
                precision_bits=64,
                data=DenseTensor.from_dense(v),
            )
            for lv, v in zip(subspace_levels, modified_values)
        },
    )


def dehierarchize(
    hierarchical_tensors: data_structures.SparseGridHierarchicalTensors,
    wavelet: BasisLike | None = None,
) -> npt.NDArray:
    """Inverse of hierarchize: synthesize the full grid of nodal values.

    Runs the inverse transform once per dimension (the inverse of the
    unidirectional principle above), so the cost is O(d * prod(extents))
    regardless of the number of subspaces. Subspaces dropped by compression
    enter as all-zero blocks. The basis defaults to the one recorded on the
    container."""
    num_dim = hierarchical_tensors.dimensions
    min_level = hierarchical_tensors.min_level
    max_level = hierarchical_tensors.max_level
    bases: tuple[Basis1D, ...] = (
        hierarchical_tensors.bases if wavelet is None else as_bases(wavelet, num_dim)
    )

    band_extents: list[dict[int, int]] = [dict() for _ in range(num_dim)]
    for stored_level, subspace in hierarchical_tensors.subspaces.items():
        for d in range(num_dim):
            band_extents[d][stored_level[d]] = subspace.extents[d]

    # per-basis band sizes as fallback for bands in which every subspace was
    # dropped
    def default_band_extent(d: int, band_level: int) -> int:
        centering = bases[d].centering
        if band_level == min_level[d]:
            return centering.num_dofs(band_level)
        return centering.num_details(band_level)

    def block_extents(level: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            band_extents[d].get(level[d], default_band_extent(d, level[d]))
            for d in range(num_dim)
        )

    # coefficient blocks in the same C-order the forward transform produces
    blocks = []
    for level in itertools.product(
        *(range(min_level[d], max_level[d] + 1) for d in range(num_dim))
    ):
        stored = hierarchical_tensors.subspaces.get(level)
        if stored is None or stored.data is None:
            blocks.append(np.zeros(block_extents(level)))
        else:
            blocks.append(stored.data.to_dense())

    # reverse order of the forward loop
    for d in reversed(range(num_dim)):
        num_bands = max_level[d] - min_level[d] + 1
        if num_bands == 1:
            continue
        blocks = [
            reconstruct_axis(blocks[start : start + num_bands], d, bases[d])
            for start in range(0, len(blocks), num_bands)
        ]

    assert len(blocks) == 1
    return blocks[0]
