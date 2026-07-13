import numpy as np

from spght.compress import compress
from spght.data_structures import (
    DenseTensor,
    SparseGridHierarchicalTensors,
    Subspace,
    TensorKind,
)
from spght.hierarchize import hierarchize
from spght.interpolate import interpolate
from spght.linearize import midpoint_coordinates_from_level


def _subspace(array) -> Subspace:
    array = np.asarray(array, dtype=np.float64)
    return Subspace(
        extents=array.shape,
        precision_bits=64,
        data=DenseTensor.from_dense(array),
    )


def test_compress_small_checker():
    nodal_values = np.array([[1.0, 0.0], [0.0, 1.0]])
    hierarchical_values = hierarchize(nodal_values)
    compressed_values = compress(hierarchical_values, only_whole_subspaces=True)
    assert compressed_values.dimensions == 2
    assert compressed_values.max_level == (1, 1)
    assert len(compressed_values.subspaces) == 2
    assert compressed_values.subspaces[(0, 0)].values == [0.5]
    assert compressed_values.subspaces[(1, 1)].values == [0.5]


def test_compress_partial_lossless_by_default():
    # epsilon=0 without only_whole_subspaces: exact-zero coefficients are
    # dropped, everything else survives -> reconstruction stays exact
    nodal_values = np.array([[1.0, 0.0], [0.0, 1.0]])
    compressed = compress(hierarchize(nodal_values))
    midpoints = midpoint_coordinates_from_level([1, 1])
    reconstructed = interpolate(midpoints, compressed)
    assert np.allclose(reconstructed, nodal_values)


def test_compress_partial_sparsifies_low_density():
    coefficients = np.full((4, 4), 0.01)
    coefficients[0, 0] = 5.0  # 1 of 16 survives -> density 1/16 < 0.1
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(3, 3),
        subspaces={(0, 0): _subspace([[1.0]]), (3, 3): _subspace(coefficients)},
    )
    compressed = compress(tensors, epsilon=0.1)
    data = compressed.subspaces[(3, 3)].data
    assert data is not None
    assert data.kind == TensorKind.LINEAR  # only compress() sparsifies
    assert data.nnz == 1
    dense = data.to_dense()
    assert dense[0, 0] == 5.0
    assert np.count_nonzero(dense) == 1


def test_compress_partial_stays_dense_at_high_density():
    coefficients = np.full((4, 4), 0.01)
    coefficients[:2, :] = 5.0  # 8 of 16 survive -> density 0.5 >= 0.1
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(3, 3),
        subspaces={(0, 0): _subspace([[1.0]]), (3, 3): _subspace(coefficients)},
    )
    compressed = compress(tensors, epsilon=0.1)
    data = compressed.subspaces[(3, 3)].data
    assert data is not None
    assert data.kind == TensorKind.FULL  # dropped coefficients zeroed in place
    assert data.nnz == 8
    assert np.allclose(data.to_dense()[:2, :], 5.0)
    assert np.all(data.to_dense()[2:, :] == 0.0)


def test_compress_max_level_elementwise():
    # kept levels (1, 0) and (0, 2): the maximum level must be the
    # elementwise max (1, 2), not the lexicographically largest key (1, 0)
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(2, 2),
        subspaces={
            (0, 0): _subspace([[1.0]]),
            (1, 0): _subspace([[1.0]]),
            (0, 2): _subspace([[1.0, 1.0]]),
            (2, 2): _subspace(np.zeros((2, 2))),  # dropped
        },
    )
    compressed = compress(tensors, only_whole_subspaces=True)
    assert set(compressed.subspaces.keys()) == {(0, 0), (1, 0), (0, 2)}
    assert compressed.max_level == (1, 2)


def test_compress_always_keeps_lmin():
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(1, 1),
        subspaces={
            (0, 0): _subspace([[0.0]]),  # all-zero, but lmin
            (1, 1): _subspace([[0.0, 0.0]]),
        },
    )
    compressed = compress(tensors, only_whole_subspaces=True, epsilon=1.0)
    assert set(compressed.subspaces.keys()) == {(0, 0)}
    assert compressed.max_level == (0, 0)


def test_compress_preserves_canonical_order():
    from spght.data_structures import subspace_order_key

    rng = np.random.default_rng(5)
    hierarchical = hierarchize(rng.random((8, 8)))
    compressed = compress(hierarchical, epsilon=0.05)
    keys = list(compressed.subspaces.keys())
    assert keys == sorted(keys, key=subspace_order_key)


def test_compress_partial_error_bounded():
    rng = np.random.default_rng(3)
    level = [5, 5, 5]
    nodal_values = rng.random((32, 32, 32))
    hierarchical = hierarchize(nodal_values)
    assert len(hierarchical.subspaces) == 6**3
    epsilon = 0.02
    compressed = compress(hierarchical, epsilon=epsilon)

    midpoints = midpoint_coordinates_from_level(level)
    reconstructed = interpolate(midpoints, compressed)
    # each subspace contributes one (Haar) coefficient per point, so every
    # dropped coefficient adds at most epsilon
    error_bound = epsilon * len(hierarchical.subspaces)
    assert np.max(np.abs(reconstructed - nodal_values)) <= error_bound
    # and something must actually have been compressed away
    def total_bytes(tensors) -> int:
        return sum(s.num_bytes for s in tensors.subspaces.values())

    assert total_bytes(compressed) < total_bytes(hierarchical)
