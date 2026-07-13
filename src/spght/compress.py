from copy import deepcopy
from dataclasses import replace
import numpy as np


from spght.data_structures import (
    SparseGridHierarchicalTensors,
    make_tensor_from_linear,
)


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
        subspaces=dict(),
    )
    for level, subspace in hierarchical_tensors.subspaces.items():
        assert subspace.data is not None
        coefficients = subspace.data.linear_values
        keep = np.abs(coefficients) > epsilon
        if sum(level) == 0:
            # always keep the lmin subspace
            compressed_tensors.subspaces[level] = deepcopy(subspace)
        elif not keep.any():
            continue
        elif keep.all() or only_whole_subspaces:
            compressed_tensors.subspaces[level] = deepcopy(subspace)
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
            compressed_tensors.subspaces[level] = replace(
                subspace, data=compressed_data
            )

    # the maximum level is the elementwise maximum over the kept subspaces
    if compressed_tensors.subspaces:
        compressed_tensors.max_level = tuple(
            max(levels) for levels in zip(*compressed_tensors.subspaces.keys())
        )
    else:
        compressed_tensors.max_level = tuple(
            0 for _ in range(compressed_tensors.dimensions)
        )
    return compressed_tensors
