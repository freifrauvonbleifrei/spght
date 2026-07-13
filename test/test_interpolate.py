# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from spght.interpolate import interpolate
from spght.hierarchize import hierarchize
from spght.linearize import extent_from_level, midpoint_coordinates_from_level


def test_interpolate_random_level0():
    for dimensionality in range(1, 5):
        nodal_values = np.ones([1] * dimensionality) * np.random.rand()
        hierarchical_tensors = hierarchize(nodal_values)
        coordinate = np.random.rand(dimensionality)
        value = interpolate(coordinate, hierarchical_tensors)
        assert np.isclose(value, nodal_values[0])


def test_interpolate_1d_level2():
    for coordinate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        nodal_values = np.array([1.0, -1.0, 1.0, -1.0])
        hierarchical_tensors = hierarchize(nodal_values)
        value = interpolate(np.array([coordinate]), hierarchical_tensors)
        if coordinate < 0.25:
            assert np.isclose(value, 1.0)
        elif coordinate < 0.5:
            assert np.isclose(value, -1.0)
        elif coordinate < 0.75:
            assert np.isclose(value, 1.0)
        else:
            assert np.isclose(value, -1.0)


def test_interpolate_1d_level3():
    for coordinate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        nodal_values = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0])
        hierarchical_tensors = hierarchize(nodal_values)
        value = interpolate(np.array([coordinate]), hierarchical_tensors)
        if coordinate < 0.125:
            assert np.isclose(value, 1.0)
        elif coordinate < 0.25:
            assert np.isclose(value, -2.0)
        elif coordinate < 0.375:
            assert np.isclose(value, 3.0)
        elif coordinate < 0.5:
            assert np.isclose(value, -4.0)
        elif coordinate < 0.625:
            assert np.isclose(value, 5.0)
        elif coordinate < 0.75:
            assert np.isclose(value, -6.0)
        elif coordinate < 0.875:
            assert np.isclose(value, 7.0)
        else:
            assert np.isclose(value, -8.0)


def test_interpolate_2d_level_1_1():
    coordinates = np.array([0.3, 0.7])
    nodal_values = np.array([[1.0, 2.0], [3.0, 4.0]])
    hierarchical_tensors = hierarchize(nodal_values)
    value = interpolate(coordinates, hierarchical_tensors)
    assert np.isclose(value, 2.0)


def test_interpolate_2d_level_2_1():
    coordinates = np.array([0.6, 0.2])
    nodal_values = np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]])
    hierarchical_tensors = hierarchize(nodal_values)
    value = interpolate(coordinates, hierarchical_tensors)
    assert np.isclose(value, 3.0)


def test_interpolate_many_2d_level_2_3():
    nodal_values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
            [9.0, 10.0, 11.0, 12.0],
            [11.0, 12.0, 13.0, 14.0],
            [13.0, 14.0, 15.0, 16.0],
            [15.0, 16.0, 17.0, 18.0],
        ]
    )
    hierarchical_tensors = hierarchize(nodal_values)
    coordinates_1d = np.array([[0.6, 0.2], [0.1, 0.9], [0.2, 0.4]])
    values = interpolate(coordinates_1d, hierarchical_tensors)
    assert np.isclose(values[0], 9.0)
    assert np.isclose(values[1], 4.0)
    assert np.isclose(values[2], 4.0)

    coordinates_2d = np.array([[[0.6, 0.2], [0.1, 0.9]], [[0.2, 0.4], [0.8, 0.8]]])
    values_2d = interpolate(coordinates_2d, hierarchical_tensors)
    assert np.isclose(values_2d[0, 0], 9.0)
    assert np.isclose(values_2d[0, 1], 4.0)
    assert np.isclose(values_2d[1, 0], 4.0)
    assert np.isclose(values_2d[1, 1], 16.0)


def test_interpolate_3d_level_1_1_1():
    # Interpolating at the cell midpoints of a full grid must reproduce the
    # nodal values exactly, in the right places. This fails if the subspaces
    # are labelled with permuted level tuples (an axis-order / indexing bug that
    # only surfaces for num_dim >= 3).
    midpoint_coordinates = midpoint_coordinates_from_level([1, 1, 1])
    nodal_values = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    hierarchical_tensors = hierarchize(nodal_values)
    interpolated_values = interpolate(midpoint_coordinates, hierarchical_tensors)
    assert np.isclose(interpolated_values, nodal_values).all()


def test_interpolate_3d_level_2_2_2():
    nodal_values = np.arange(64).reshape((4, 4, 4)).astype(np.float32)
    hierarchical_tensors = hierarchize(nodal_values)

    # Reconstructing at every cell midpoint must return the nodal values in the
    # correct positions -- the strong check for the ndim >= 3 indexing bug.
    midpoint_coordinates = midpoint_coordinates_from_level([2, 2, 2])
    interpolated_values = interpolate(midpoint_coordinates, hierarchical_tensors)
    assert np.isclose(interpolated_values, nodal_values).all()

    # And a single interior point: piecewise-constant (Haar) reconstruction
    # returns the value of the cell containing it, floor([0.6, 0.2, 0.8] * 4).
    value = interpolate(np.array([0.6, 0.2, 0.8]), hierarchical_tensors)
    assert np.isclose(value, nodal_values[2, 0, 3])


def test_interpolate_fine_scale_random():
    # Reconstruction at cell midpoints of a fine grid must reproduce the nodal
    # values to (near) machine precision. Summing many subspaces in single
    # precision would blow the tolerance -- this guards that numerical error.
    np.random.seed(0)
    for level in ([10], [5, 5], [4, 4, 4]):
        dimensionality = len(level)
        extents = [extent_from_level(lvl) for lvl in level]
        nodal_values = np.random.rand(*extents).astype(np.float64)
        hierarchical_tensors = hierarchize(nodal_values)
        midpoints = midpoint_coordinates_from_level(level)
        values = interpolate(midpoints, hierarchical_tensors)
        max_error = np.max(np.abs(values - nodal_values))
        assert max_error < 1e-9, (
            f"max error {max_error} exceeds tolerance for dimensionality "
            f"{dimensionality}"
        )


def test_interpolate_ignores_reserved_quantization_fields():
    from spght.data_structures import SparseGridHierarchicalTensors, Subspace
    from spght.tensor import DenseTensor

    # the quantization fields are reserved: interpolation returns the stored
    # coefficient unchanged
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(0, 0),
        subspaces={
            (0, 0): Subspace(
                extents=(1, 1),
                precision_bits=64,
                data=DenseTensor.from_dense(np.array([[30.0]])),
                quantization_scale=0.1,
                quantization_offset=-1.0,
                quantization_parameter=2.5,
            )
        },
    )
    value = interpolate(np.array([0.4, 0.7]), tensors)
    assert np.isclose(value, 30.0)


def test_interpolate_min_level_reconstruction_exact():
    # stopping the cascade at min_level must not change what the hierarchy
    # represents: midpoint reconstruction stays exact
    rng = np.random.default_rng(7)
    for shape, min_level in [((16,), 2), ((8, 8), (1, 2)), ((4, 4, 4), 1)]:
        nodal_values = rng.random(shape)
        hierarchical_tensors = hierarchize(nodal_values, min_level=min_level)
        level = [int(np.log2(extent)) for extent in shape]
        midpoints = midpoint_coordinates_from_level(level)
        values = interpolate(midpoints, hierarchical_tensors)
        assert np.allclose(values, nodal_values)
