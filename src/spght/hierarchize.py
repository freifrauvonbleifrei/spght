# Hierarchize a multi-dimensional function on a structured grid using the unidirectional principle.

import itertools
import numpy as np
import numpy.typing as npt
import spght.data_structures as data_structures
import pywt
from typing import Sequence

from spght.linearize import level_from_extent
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

    # construct a matching list of subspace levels: tensor product of 1D levels
    # from 0 to level[d] for each dimension d. The order must match the order in
    # which `modified_values` was built by the nested wavedec loop above, i.e.
    # C-order.
    subspace_levels = list(
        itertools.product(*(range(level[d] + 1) for d in range(num_dim)))
    )

    return data_structures.SparseGridHierarchicalTensors(
        dimensions=num_dim,
        max_level=tuple(level),
        subspaces={
            # construction from a full array always yields dense (linear)
            # storage; sparsification only happens in compress()
            tuple(lv): data_structures.Subspace(
                extents=v.shape,
                precision_bits=64,
                data=data_structures.DenseTensor.from_dense(v),
            )
            for lv, v in zip(subspace_levels, modified_values)
        },
    )
