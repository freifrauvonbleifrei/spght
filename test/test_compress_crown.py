# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import itertools

import numpy as np
import pytest

from spght import (
    DenseTensor,
    IntervalTensor,
    SparseGridHierarchicalTensors,
    SparseTensor,
    Subspace,
    compress,
    haar_basis,
    hat_basis,
    hierarchize,
    interpolate,
)
from spght.data_structures import subspace_order_key


def _block(values):
    data = DenseTensor.from_dense(np.asarray(values, dtype=float))
    return Subspace(extents=data.shape, precision_bits=64, data=data)


def _overlap_oracle(tensors, epsilon, whole):
    """Exhaustive geometric comparison; no parent mappings or graph traversal."""
    retained = []
    result = {}
    for level in sorted(tensors.subspaces, key=subspace_order_key, reverse=True):
        subspace = tensors.subspaces[level]
        array = (
            np.zeros(subspace.extents)
            if subspace.data is None
            else subspace.data.to_dense()
        )
        output = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            value = array[index]
            lower = np.array(index) / array.shape
            upper = (np.array(index) + 1) / array.shape
            blocked = any(
                all(a <= b for a, b in zip(level, fine_level))
                and level != fine_level
                and np.all(np.maximum(lower, lo) < np.minimum(upper, hi))
                for fine_level, lo, hi in retained
            )
            if level == tensors.min_level or abs(value) > epsilon or blocked:
                output[index] = value
        if whole and np.any(output):
            output = array.copy()
        result[level] = output
        for index in zip(*np.nonzero(output)):
            retained.append(
                (
                    level,
                    np.array(index) / array.shape,
                    (np.array(index) + 1) / array.shape,
                )
            )
    return result


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
@pytest.mark.parametrize("storage", [DenseTensor, SparseTensor, IntervalTensor])
@pytest.mark.parametrize("whole", [False, True])
@pytest.mark.parametrize("minimum", [(0, 0), (1, 2), (0, 1, 0)])
def test_crown_matches_support_overlap_oracle(order, storage, whole, minimum):
    rng = np.random.default_rng(918)
    maximum = tuple(m + 2 for m in minimum)
    blocks = {}
    for level in itertools.product(
        *(range(m, n + 1) for m, n in zip(minimum, maximum))
    ):
        shape = tuple(2 ** (l if l == m else l - 1) for l, m in zip(level, minimum))
        # Include absent intermediate subspaces and explicitly EMPTY blocks.
        if level != minimum and rng.random() < 0.2:
            continue
        values = rng.choice([0.0, 0.01, -0.1, 0.1, 1.0], size=shape)
        data = storage.from_dense(values, order=order)
        blocks[level] = Subspace(
            extents=shape,
            precision_bits=64,
            data=None if level != minimum and rng.random() < 0.15 else data,
        )
    tensors = SparseGridHierarchicalTensors(
        dimensions=len(minimum),
        min_level=minimum,
        max_level=maximum,
        subspaces=blocks,
        metadata={"field_name": "crown test"},
    )
    before = {l: s.data.to_dense() for l, s in blocks.items() if s.data is not None}
    expected = _overlap_oracle(tensors, epsilon=0.1, whole=whole)
    result = compress(
        tensors, epsilon=0.1, structure="crown", only_whole_subspaces=whole
    )
    repeated = compress(
        result, epsilon=0.1, structure="crown", only_whole_subspaces=whole
    )
    independent = compress(tensors, epsilon=0.1, only_whole_subspaces=whole)
    assert result.min_level == minimum
    assert result.metadata == tensors.metadata
    assert result.bases == tensors.bases
    assert result.subspaces[minimum] is tensors.subspaces[minimum]
    assert list(result.subspaces) == sorted(result.subspaces, key=subspace_order_key)
    assert result.max_level == tuple(map(max, zip(*result.subspaces)))
    for level, array in expected.items():
        stored = result.subspaces.get(level)
        actual = (
            np.zeros_like(array)
            if stored is None or stored.data is None
            else stored.data.to_dense()
        )
        np.testing.assert_array_equal(actual, array)
        if stored is not None and stored.data is not None:
            assert stored.data.order == order
            np.testing.assert_array_equal(
                repeated.subspaces[level].data.to_dense(), actual
            )
        ordinary = independent.subspaces.get(level)
        if ordinary is not None and ordinary.data is not None:
            surviving = ordinary.data.to_dense() != 0
            np.testing.assert_array_equal(
                actual[surviving], ordinary.data.to_dense()[surviving]
            )
    for level, array in before.items():
        np.testing.assert_array_equal(tensors.subspaces[level].data.to_dense(), array)


def test_crown_crosses_missing_levels_and_zero_entries():
    tensors = SparseGridHierarchicalTensors(
        dimensions=1,
        max_level=(4,),
        subspaces={
            (0,): _block([2]),
            (1,): _block([0.01]),
            (2,): _block([0, 0.02]),
            # The intermediate level 3 is completely absent.
            (4,): _block([1, 0, 0, 0, 0, 0, 0, 0]),
        },
    )
    result = compress(tensors, epsilon=0.1, structure="crown")
    assert result.subspaces[(1,)] is tensors.subspaces[(1,)]
    assert (2,) not in result.subspaces
    assert (3,) not in result.subspaces
    np.testing.assert_array_equal(
        result.subspaces[(4,)].data.to_dense(), [1, 0, 0, 0, 0, 0, 0, 0]
    )


def test_crown_whole_block_retention_protects_other_branch():
    tensors = SparseGridHierarchicalTensors(
        dimensions=1,
        min_level=(1,),
        max_level=(4,),
        subspaces={
            (1,): _block([0, 0]),
            (2,): _block([0.01, 0.02]),
            (3,): _block([0.03, 0, 0, 0.04]),
            (4,): _block([1, 0, 0, 0, 0, 0, 0, 0]),
        },
    )
    pointwise = compress(tensors, epsilon=0.1, structure="crown")
    np.testing.assert_array_equal(pointwise.subspaces[(2,)].data.to_dense(), [0.01, 0])
    whole = compress(tensors, epsilon=0.1, structure="crown", only_whole_subspaces=True)
    for level in tensors.subspaces:
        assert whole.subspaces[level] is tensors.subspaces[level]


@pytest.mark.parametrize("epsilon", [0.0, 1e6])
def test_crown_reconstruction_and_scaling_protection(epsilon):
    rng = np.random.default_rng(31)
    values = rng.random((8, 16))
    tensors = hierarchize(values, min_level=(1, 2))
    result = compress(tensors, epsilon=epsilon, structure="crown")
    coords = np.stack(
        np.meshgrid(
            (np.arange(8) + 0.5) / 8, (np.arange(16) + 0.5) / 16, indexing="ij"
        ),
        axis=-1,
    )
    if epsilon == 0:
        np.testing.assert_allclose(interpolate(coords, result), values)
    else:
        assert list(result.subspaces) == [(1, 2)]
        assert result.max_level == (1, 2)


def test_crown_rejects_unsupported_basis_and_invalid_geometry():
    tensors = hierarchize(np.ones((5,)), wavelet=hat_basis())
    with pytest.raises(ValueError, match="Haar"):
        compress(tensors, structure="crown")
    malformed = SparseGridHierarchicalTensors(
        dimensions=1, max_level=(2,), subspaces={(2,): _block([1, 2, 3])}
    )
    with pytest.raises(ValueError, match="geometry"):
        compress(malformed, structure="crown")
    # Default compression retains its existing behavior for such containers.
    compress(malformed)


def test_crown_identifies_basis_by_steps_not_label():
    basis = haar_basis()
    renamed = replace(basis, scheme=replace(basis.scheme, name="custom label"))
    tensors = hierarchize(np.arange(8.0), wavelet=renamed)
    compress(tensors, structure="crown")
    fake = replace(basis, scheme=replace(basis.scheme, steps=hat_basis().scheme.steps))
    tensors.bases = (fake,)
    with pytest.raises(ValueError, match="Haar"):
        compress(tensors, structure="crown")


@pytest.mark.parametrize("epsilon", [-1, np.nan, np.inf])
def test_crown_rejects_invalid_epsilon(epsilon):
    with pytest.raises(ValueError, match="epsilon"):
        compress(hierarchize(np.ones(4)), epsilon=epsilon, structure="crown")


def test_crown_preserves_nonfinite_values_and_ancestors():
    tensors = SparseGridHierarchicalTensors(
        dimensions=1,
        max_level=(2,),
        subspaces={
            (0,): _block([0]),
            (1,): _block([0.01]),
            (2,): _block([np.nan, np.inf]),
        },
    )
    result = compress(tensors, epsilon=1, structure="crown")
    assert result.subspaces[(1,)] is tensors.subspaces[(1,)]
    assert result.subspaces[(2,)] is tensors.subspaces[(2,)]


def test_compress_rejects_unknown_structure():
    with pytest.raises(ValueError, match="structure"):
        compress(hierarchize(np.ones(4)), structure="typo")
