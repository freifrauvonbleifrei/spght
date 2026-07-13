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
    minimum_levels = min_level
    if isinstance(min_level, int):
        minimum_levels = [min_level] * num_dim
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
