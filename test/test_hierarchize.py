import numpy as np

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
    assert np.allclose(
        result_coarser.subspaces[(0,)].data, result.subspaces[(0,)].data
    )
    assert np.allclose(
        result_coarser.subspaces[(1,)].data, result.subspaces[(1,)].data
    )


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
