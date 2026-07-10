# Hierarchize a multi-dimensional function on a structured grid using the unidirectional principle.

import numpy as np
import numpy.typing as npt
import spght.data_structures as data_structures
import pywt
from typing import Sequence

from spght.linearization import level_from_extent
from spght.wavelets import half_haar


def hierarchize(
    nodal_values: npt.NDArray,
    deviate_from_power_of_two: int | Sequence[int] = 0,
    wavelet: pywt.Wavelet = half_haar,
) -> data_structures.SparseGridHierarchicalTensors:
    num_dim = nodal_values.ndim
    level: npt.NDArray = np.ndarray(num_dim, dtype=int)
    for d in range(num_dim):
        if isinstance(deviate_from_power_of_two, int):
            level[d] = level_from_extent(
                nodal_values.shape[d] - deviate_from_power_of_two
            )
        else:
            level[d] = level_from_extent(
                nodal_values.shape[d] - deviate_from_power_of_two[d]
            )

    modified_values = [nodal_values.copy()]
    for d in range(num_dim):
        # Apply the Haar wavelet transform along each dimension
        updated_values = []
        for slices in modified_values:
            updated_values.extend(pywt.wavedec(slices, wavelet, axis=d))
            # TODO add lmin
        modified_values = updated_values

    # construct a matching list of subspace levels: tensor product of 1D levels from 0 to level[d] for each dimension d
    subspace_levels_per_dim = []
    for d in range(num_dim):
        subspace_levels_per_dim.append(list(range(level[d] + 1)))
    subspace_levels = np.array(np.meshgrid(*subspace_levels_per_dim)).T.reshape(
        -1, num_dim
    )

    return data_structures.SparseGridHierarchicalTensors(
        dimensions=num_dim,
        max_level=tuple(level),
        subspaces={
            tuple(lv): data_structures.Subspace(
                extents=v.shape,
                precision_bits=64,
                values=v,  # modified_values.tobytes(),
            )
            for lv, v in zip(subspace_levels, modified_values)
        },
    )
