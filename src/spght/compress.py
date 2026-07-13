from copy import deepcopy
import numpy as np


from spght.data_structures import SparseGridHierarchicalTensors


def compress(
    hierarchical_tensors: SparseGridHierarchicalTensors,
    only_whole_subspaces: bool = False,
    epsilon: float = 0.0,
) -> SparseGridHierarchicalTensors:
    compressed_tensors = SparseGridHierarchicalTensors(
        dimensions=hierarchical_tensors.dimensions,
        max_level=hierarchical_tensors.max_level,
        subspaces=dict(),
    )
    for level, subspace in hierarchical_tensors.subspaces.items():
        assert subspace.data is not None
        coefficients = subspace.data.linear_values
        if np.all(np.abs(coefficients) <= epsilon):
            continue
        elif not only_whole_subspaces and np.any(np.abs(coefficients) < epsilon):
            raise NotImplementedError(
                "Partial subspace compression is not implemented yet."
            )
        else:
            compressed_tensors.subspaces[level] = deepcopy(subspace)
    return compressed_tensors
