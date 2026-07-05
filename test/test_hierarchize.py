import numpy as np
from icecream import ic

from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.hierarchize import hierarchize


def test_hierarchize_haar_1d():
    nodal_values = np.array([1.0, 2.0, 3.0, 4.0])
    result = hierarchize(nodal_values)
    ic(result)
    # assert isinstance(result, SparseGridHierarchicalTensors) #??
    assert result.dimensions == 1
    assert result.max_level == (2,)
    assert result.subspaces.keys() == {(0,), (1,), (2,)}
    assert result.subspaces[(0,)].extents == (1,)
    assert result.subspaces[(0,)].precision_bits == 64
    assert np.allclose(result.subspaces[(0,)].values, np.array([5.0]))
    assert result.subspaces[(1,)].extents == (1,)
    assert result.subspaces[(1,)].precision_bits == 64
    assert np.allclose(result.subspaces[(1,)].values, np.array([-2.0]))
    assert result.subspaces[(2,)].extents == (2,)
    assert result.subspaces[(2,)].precision_bits == 64
    assert np.allclose(
        result.subspaces[(2,)].values, np.array([-0.70710678, -0.70710678])
    )


def test_hierarchize_haar_2d():
    nodal_values = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    result = hierarchize(nodal_values)
    # assert isinstance(result, SparseGridHierarchicalTensors)
    assert result.dimensions == 2
    assert result.max_level == (1, 2)
    assert len(result.subspaces) == 6
    assert result.subspaces[(0, 0)].extents == (1, 1)
    assert result.subspaces[(0, 0)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 0)].values, np.array([[9.89949494]]))
    assert result.subspaces[(0, 1)].extents == (1, 1)
    assert result.subspaces[(0, 1)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 1)].values, np.array([[-2.82842712]]))
    assert result.subspaces[(0, 2)].extents == (1, 2)
    assert result.subspaces[(0, 2)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 2)].values, np.array([[-1.0, -1.0]]))
    assert result.subspaces[(1, 1)].extents == (1, 1)
    assert result.subspaces[(1, 1)].precision_bits == 64
    assert np.allclose(
        result.subspaces[(1, 1)].values, np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    assert result.subspaces[(1, 2)].extents == (1, 2)
    assert result.subspaces[(1, 2)].precision_bits == 64
    assert np.allclose(
        result.subspaces[(1, 2)].values, np.array([[0.0, 0.0], [0.0, 0.0]])
    )
    assert result.subspaces[(0, 2)].extents == (1, 2)
    assert result.subspaces[(0, 2)].precision_bits == 64
    assert np.allclose(result.subspaces[(0, 2)].values, np.array([[-1.0, -1.0]]))
