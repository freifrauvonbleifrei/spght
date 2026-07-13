# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

# Hierarchize a multi-dimensional function on a structured grid using the unidirectional principle.

import itertools
import numpy as np
import numpy.typing as npt
import pywt
from typing import Sequence

import spght.data_structures as data_structures
from spght.linearize import level_from_extent
from spght.tensor import DenseTensor
from spght.wavelets import half_haar


def hierarchize(
    nodal_values: npt.NDArray,
    wavelet: pywt.Wavelet = half_haar,
    min_level: int | Sequence[int] = 0,
) -> data_structures.SparseGridHierarchicalTensors:
    """Decompose nodal values into hierarchical subspaces."""
    num_dim = nodal_values.ndim
    if isinstance(min_level, int):
        minimum_levels = [min_level] * num_dim
    else:
        minimum_levels = list(min_level)
    level: npt.NDArray = np.ndarray(num_dim, dtype=int)
    for d in range(num_dim):
        level[d] = level_from_extent(nodal_values.shape[d])
        if not 0 <= minimum_levels[d] <= level[d]:
            raise ValueError(
                f"min_level {minimum_levels} must be between 0 and the "
                f"maximum level {tuple(level[: d + 1])} in every dimension"
            )

    modified_values = [nodal_values.copy()]
    for d in range(num_dim):
        # Apply the wavelet transform along each dimension, stopping the
        # cascade at min_level
        num_levels = int(level[d]) - minimum_levels[d]
        updated_values = []
        for slices in modified_values:
            if num_levels == 0:
                updated_values.append(slices)
            else:
                updated_values.extend(
                    pywt.wavedec(slices, wavelet, axis=d, level=num_levels)
                )
        modified_values = updated_values

    # construct a matching list of subspace levels: tensor product of 1D levels
    # from min_level[d] to level[d] for each dimension d. The order must match
    # the order in which `modified_values` was built by the nested wavedec loop
    # above, i.e. C-order (wavedec returns the coarsest block first).
    subspace_levels = list(
        itertools.product(
            *(range(minimum_levels[d], level[d] + 1) for d in range(num_dim))
        )
    )

    return data_structures.SparseGridHierarchicalTensors(
        dimensions=num_dim,
        max_level=tuple(level),
        min_level=tuple(minimum_levels),
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
    wavelet: pywt.Wavelet = half_haar,
) -> npt.NDArray:
    """Inverse of hierarchize: synthesize the full grid of nodal values."""
    num_dim = hierarchical_tensors.dimensions
    min_level = hierarchical_tensors.min_level
    max_level = hierarchical_tensors.max_level

    band_extents: list[dict[int, int]] = [dict() for _ in range(num_dim)]
    for stored_level, subspace in hierarchical_tensors.subspaces.items():
        for d in range(num_dim):
            band_extents[d][stored_level[d]] = subspace.extents[d]

    # dyadic sizes as fallback for bands in which every subspace was dropped
    def block_extents(level: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(
            band_extents[d].get(
                level[d],
                2 ** level[d] if level[d] == min_level[d] else 2 ** (level[d] - 1),
            )
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
            pywt.waverec(blocks[start : start + num_bands], wavelet, axis=d)
            for start in range(0, len(blocks), num_bands)
        ]

    assert len(blocks) == 1
    return blocks[0]
