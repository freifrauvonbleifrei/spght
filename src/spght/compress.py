# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import replace
import numpy as np


from spght.data_structures import SparseGridHierarchicalTensors

from spght.tensor import make_tensor_from_linear


def compress(
    hierarchical_tensors: SparseGridHierarchicalTensors,
    only_whole_subspaces: bool = False,
    epsilon: float = 0.0,
    density_threshold: float = 0.5,
) -> SparseGridHierarchicalTensors:
    """Drop hierarchical coefficients with |coefficient| <= epsilon."""
    compressed_tensors = SparseGridHierarchicalTensors(
        dimensions=hierarchical_tensors.dimensions,
        max_level=hierarchical_tensors.max_level,
        min_level=hierarchical_tensors.min_level,
        bases=hierarchical_tensors.bases,
        subspaces=dict(),
    )
    for level, subspace in hierarchical_tensors.subspaces.items():
        assert subspace.data is not None
        coefficients = subspace.data.linear_values
        keep = np.abs(coefficients) > epsilon
        if level == hierarchical_tensors.min_level:
            # always keep the all-scaling lmin subspace
            compressed_tensors.add_subspace(level, deepcopy(subspace))
        elif not keep.any():
            continue
        elif keep.all() or only_whole_subspaces:
            compressed_tensors.add_subspace(level, deepcopy(subspace))
        else:
            # partial compression: keep only the surviving coefficients
            keys = subspace.data.linear_indices[keep]
            compressed_data = make_tensor_from_linear(
                keys,
                coefficients[keep],
                subspace.data.shape,
                order=subspace.data.order,
                density_threshold=density_threshold,
            )
            compressed_tensors.add_subspace(
                level, replace(subspace, data=compressed_data)
            )

    # the maximum level is the elementwise maximum over the kept subspaces
    if compressed_tensors.subspaces:
        compressed_tensors.max_level = tuple(
            max(levels) for levels in zip(*compressed_tensors.subspaces.keys())
        )
    else:
        compressed_tensors.max_level = tuple(compressed_tensors.min_level)
    return compressed_tensors
