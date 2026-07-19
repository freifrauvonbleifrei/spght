# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import io

import numpy as np
import pytest

from spght.compress import compress
from spght.data_structures import SparseGridHierarchicalTensors
from spght.hierarchize import dehierarchize, hierarchize
from spght.interpolate import interpolate
from spght.linearize import midpoint_coordinates_from_level
from spght.tensor import DenseTensor, SparseTensor
from spght.wavelets import hat_basis

ORDERS = ("C", "F", "ZC", "ZF")


@pytest.mark.parametrize("order", ORDERS)
def test_dense_with_order_matches_from_dense(order):
    array = np.random.default_rng(0).random((2, 4, 3))  # non-pow-2: bisection
    tensor = DenseTensor.from_dense(array, order="C")
    relinearized = tensor.with_order(order)
    assert relinearized.order == order
    assert relinearized == DenseTensor.from_dense(array, order=order)
    assert np.array_equal(relinearized.to_dense(), array)


@pytest.mark.parametrize("order", ORDERS)
def test_sparse_with_order_matches_from_dense(order):
    array = np.zeros((4, 8))
    array[1, 3] = 2.0
    array[3, 5] = -1.0
    tensor = SparseTensor.from_dense(array, order="ZF")
    relinearized = tensor.with_order(order)
    assert relinearized.order == order
    assert relinearized.nnz == 2  # keys are remapped, never materialized
    assert relinearized == SparseTensor.from_dense(array, order=order)


def test_with_order_shares_when_unchanged():
    tensor = DenseTensor.from_dense(np.ones((2, 2)), order="ZC")
    assert tensor.with_order("ZC") is tensor
    sparse = SparseTensor.from_dense(np.eye(2), order="F")
    assert sparse.with_order("F") is sparse


def test_with_order_round_trip_identity():
    array = np.random.default_rng(1).random((4, 2, 8))
    tensor = DenseTensor.from_dense(array, order="C")
    round_tripped = tensor.with_order("ZC").with_order("F").with_order("C")
    assert round_tripped == tensor


def test_relinearize_container_in_place():
    hierarchical = hierarchize(np.random.default_rng(2).random((8, 8)))
    reference = {
        level: subspace.data.to_dense()
        for level, subspace in hierarchical.subspaces.items()
    }
    result = hierarchical.relinearize("ZC")
    assert result is hierarchical  # in place, returned for chaining
    for level, subspace in hierarchical.subspaces.items():
        assert subspace.data.order == "ZC"
        assert np.array_equal(subspace.data.to_dense(), reference[level])


@pytest.mark.parametrize("order", ORDERS)
def test_interpolate_invariant_under_relinearization(order):
    nodal_values = np.random.default_rng(3).random((8, 4, 8))
    hierarchical = hierarchize(nodal_values)
    coordinates = np.random.default_rng(4).random((37, 3))
    expected = interpolate(coordinates, hierarchical)
    hierarchical.relinearize(order)
    assert np.array_equal(interpolate(coordinates, hierarchical), expected)


@pytest.mark.parametrize("order", ORDERS)
def test_dehierarchize_invariant_under_relinearization(order):
    nodal_values = np.random.default_rng(5).random((8, 8))
    hierarchical = hierarchize(nodal_values).relinearize(order)
    assert np.allclose(dehierarchize(hierarchical), nodal_values)


def test_interpolate_invariant_with_hat_basis():
    # vertex-centered hat basis: 2^l + 1 extents exercise the non-power-of-
    # two bisection Z-order
    nodal_values = np.random.default_rng(6).random((9, 9))
    hierarchical = hierarchize(nodal_values, wavelet=hat_basis())
    coordinates = np.random.default_rng(7).random((23, 2))
    expected = interpolate(coordinates, hierarchical)
    hierarchical.relinearize("ZC")
    assert np.array_equal(interpolate(coordinates, hierarchical), expected)


def test_interpolate_invariant_after_compress():
    # sparse (POINTWISE) subspaces take the key-remapping path
    hierarchical = hierarchize(np.random.default_rng(8).random((16, 16)))
    compressed = compress(hierarchical, epsilon=0.05)
    assert any(s.data.is_sparse for s in compressed.subspaces.values())
    midpoints = midpoint_coordinates_from_level([4, 4])
    expected = interpolate(midpoints, compressed)
    compressed.relinearize("ZF")
    assert np.array_equal(interpolate(midpoints, compressed), expected)


@pytest.mark.parametrize("order", ORDERS)
def test_relinearized_order_conserved_on_disk(order):
    hierarchical = hierarchize(np.random.default_rng(9).random((8, 8)))
    hierarchical.relinearize(order)
    buffer = io.BytesIO()
    hierarchical.write(buffer)
    buffer.seek(0)
    read_back = SparseGridHierarchicalTensors.read(buffer)
    for level, subspace in read_back.subspaces.items():
        assert subspace.order == order
        assert subspace.data == hierarchical.subspaces[level].data


@pytest.mark.parametrize("order", ORDERS)
def test_interpolate_with_interval_tensors(order):
    from spght.tensor import IntervalTensor

    nodal_values = np.random.default_rng(10).random((8, 8))
    hierarchical = hierarchize(nodal_values)
    coordinates = np.random.default_rng(11).random((17, 2))
    expected = interpolate(coordinates, hierarchical)
    # replace every subspace's storage by an IntervalTensor in `order`
    from dataclasses import replace

    for level, subspace in hierarchical.subspaces.items():
        hierarchical.subspaces[level] = replace(
            subspace,
            data=IntervalTensor.from_dense(subspace.data.to_dense(), order=order),
        )
    assert np.array_equal(interpolate(coordinates, hierarchical), expected)
    assert np.allclose(dehierarchize(hierarchical), nodal_values)
