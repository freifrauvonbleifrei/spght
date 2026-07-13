# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from spght.hierarchize import hierarchize


def test_hierarchize_haar_1d():
    nodal_values = np.array([1.0, 2.0, 3.0, 4.0])
    result = hierarchize(nodal_values)
    # assert isinstance(result, SparseGridHierarchicalTensors) #??
    assert result.dimensions == 1
    assert result.max_level == (2,)
    assert result.subspaces.keys() == {(0,), (1,), (2,)}
    assert result.subspaces[(0,)].extents == (1,)
    assert result.subspaces[(0,)].precision_bits == 64
    assert np.allclose(result.subspaces[(0,)].data, np.array([2.5]))
    assert result.subspaces[(1,)].extents == (1,)
    assert result.subspaces[(1,)].precision_bits == 64
    assert np.allclose(result.subspaces[(1,)].data, np.array([-1.0]))
    assert result.subspaces[(2,)].extents == (2,)
    assert result.subspaces[(2,)].precision_bits == 64
    assert np.allclose(result.subspaces[(2,)].data, np.array([-0.5, -0.5]))

    nodal_values_coarser = np.array([1.5, 3.5])
    result_coarser = hierarchize(nodal_values_coarser)
    assert np.allclose(result_coarser.subspaces[(0,)].data, result.subspaces[(0,)].data)
    assert np.allclose(result_coarser.subspaces[(1,)].data, result.subspaces[(1,)].data)


def test_hierarchize_haar_2d():
    nodal_values = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    result = hierarchize(nodal_values)
    assert result.dimensions == 2
    assert result.max_level == (1, 2)
    assert len(result.subspaces) == 6
    assert result.subspaces[(0, 0)].extents == (1, 1)
    assert result.subspaces[(0, 0)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 0)].data, np.array([[3.5]]))
    assert result.subspaces[(0, 1)].extents == (1, 1)
    assert result.subspaces[(0, 1)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 1)].data, np.array([[-1.0]]))
    assert result.subspaces[(0, 2)].extents == (1, 2)
    assert result.subspaces[(0, 2)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 2)].data, np.array([[-0.5, -0.5]]))
    assert result.subspaces[(1, 1)].extents == (1, 1)
    assert result.subspaces[(1, 1)].precision_bits == 64
    assert np.allclose(
        result.subspaces[(1, 1)].data, np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    assert result.subspaces[(1, 2)].extents == (1, 2)
    assert result.subspaces[(1, 2)].precision_bits == 64
    assert np.allclose(
        result.subspaces[(1, 2)].data, np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    assert result.subspaces[(0, 2)].extents == (1, 2)
    assert result.subspaces[(0, 2)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 2)].data, np.array([[-0.5, -0.5]]))
    result_coarser = hierarchize(np.array([[1.5, 3.5], [3.5, 5.5]]))
    assert len(result_coarser.subspaces) == 4
    assert np.allclose(
        result_coarser.subspaces[(0, 0)].data, result.subspaces[(0, 0)].data
    )
    assert np.allclose(
        result_coarser.subspaces[(0, 1)].data, result.subspaces[(0, 1)].data
    )
    assert np.allclose(
        result_coarser.subspaces[(1, 0)].data, result.subspaces[(1, 0)].data
    )
    assert np.allclose(
        result_coarser.subspaces[(1, 1)].data, result.subspaces[(1, 1)].data
    )


def test_hierarchize_min_level():
    nodal_values = np.arange(32, dtype=np.float64).reshape(8, 4)
    result = hierarchize(nodal_values, min_level=(1, 2))
    assert result.min_level == (1, 2)
    assert result.max_level == (3, 2)
    # dimension 0 has levels 1 (scaling), 2, 3; dimension 1 only level 2 (scaling)
    assert set(result.subspaces.keys()) == {(1, 2), (2, 2), (3, 2)}
    # the scaling slot has extent 2**min_level, detail level l has 2**(l - 1);
    # dimension 1 is never transformed, so it keeps its full extent
    assert result.subspaces[(1, 2)].extents == (2, 4)
    assert result.subspaces[(2, 2)].extents == (2, 4)
    assert result.subspaces[(3, 2)].extents == (4, 4)
    # the all-scaling subspace holds the means of each half along dimension 0
    assert np.allclose(
        result.subspaces[(1, 2)].data,
        np.stack([nodal_values[:4].mean(axis=0), nodal_values[4:].mean(axis=0)]),
    )


def test_hierarchize_min_level_equal_to_max_is_identity():
    nodal_values = np.random.default_rng(6).random((4, 4))
    result = hierarchize(nodal_values, min_level=2)
    # no decomposition at all: a single all-scaling subspace with the nodal values
    assert set(result.subspaces.keys()) == {(2, 2)}
    assert np.allclose(result.subspaces[(2, 2)].data, nodal_values)


def test_hierarchize_min_level_out_of_range_raises():
    nodal_values = np.zeros((4, 4))
    with pytest.raises(ValueError):
        hierarchize(nodal_values, min_level=3)
    with pytest.raises(ValueError):
        hierarchize(nodal_values, min_level=-1)
