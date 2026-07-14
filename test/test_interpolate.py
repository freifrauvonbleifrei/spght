# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import numpy as np
import pytest

from spght.compress import compress
from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.interpolate import (
    interpolate,
    _reconstruct_subspace_and_evaluate,
    interpolate_subspace,
)
from spght.hierarchize import hierarchize, dehierarchize
from spght.linearize import extent_from_level, midpoint_coordinates_from_level
from spght.tensor import DenseTensor
from spght.wavelets import cdf_2_2_basis, hat_basis, half_haar, cubic_basis


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


def test_fast_path_matches_reconstruction():

    rng = np.random.default_rng(9)
    for shape, min_level in [((8,), 0), ((8, 4), (1, 0)), ((4, 4, 4), 1)]:
        tensors = compress(
            hierarchize(rng.random(shape), min_level=min_level), epsilon=0.05
        )
        coordinates = rng.random((32, len(shape)))
        coordinates[0] = 1.0
        coordinates[1] = 0.0
        for level, subspace in tensors.subspaces.items():
            scaling = tensors.scaling_dimensions(level)
            fast = interpolate_subspace(scaling, coordinates, subspace)
            reference = _reconstruct_subspace_and_evaluate(
                scaling, coordinates, subspace
            )
            assert np.array_equal(fast, reference), (shape, level)


def test_interpolate_defaults_to_recorded_bases():
    x = np.linspace(0.0, 1.0, 17)
    values = np.sin(2.0 * np.pi * x) + 2.0
    coordinates = np.linspace(0.0, 1.0, 41).reshape(-1, 1)
    for basis in [hat_basis(), cdf_2_2_basis()]:
        tensors = hierarchize(values, wavelet=basis)
        # hat/CDF(2,2) contributions are piecewise linear, so the exact
        # reconstruction is the linear interpolant of the nodal values
        expected = np.interp(coordinates[:, 0], x, dehierarchize(tensors))
        result = interpolate(coordinates, tensors)  # no wavelet argument
        assert np.allclose(result, expected, atol=1e-12), basis.scheme.name


def test_interpolate_recorded_bases_2d_at_nodes():
    rng = np.random.default_rng(20)
    nodal_values = rng.normal(size=(9, 17))
    tensors = hierarchize(nodal_values, wavelet=hat_basis())
    node_x = np.linspace(0.0, 1.0, 9)
    node_y = np.linspace(0.0, 1.0, 17)
    coordinates = np.stack(np.meshgrid(node_x, node_y, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    values = np.asarray(interpolate(coordinates, tensors)).reshape(9, 17)
    assert np.allclose(values, nodal_values, atol=1e-11)


def test_interpolate_default_matches_explicit_haar():
    rng = np.random.default_rng(21)
    tensors = hierarchize(rng.random((8, 8)))
    coordinates = rng.random((32, 2))
    explicit = np.zeros(coordinates.shape[0])
    for level, subspace in tensors.subspaces.items():
        explicit += interpolate_subspace(
            tensors.scaling_dimensions(level), coordinates, subspace, wavelet=half_haar
        )
    assert np.array_equal(interpolate(coordinates, tensors), explicit)


def test_interpolate_cubic_raises_not_implemented():
    x = np.linspace(0.0, 1.0, 17)
    tensors = hierarchize(x * x, wavelet=cubic_basis())
    with pytest.raises(NotImplementedError, match="direct evaluation"):
        interpolate(np.array([[0.3]]), tensors)


def test_interpolate_at_upper_boundary():
    # coordinate 1.0 is explicitly allowed by the validation and belongs
    # to the last cell; anything beyond the domain still raises

    nodal_values = np.arange(8.0)
    tensors = hierarchize(nodal_values)
    assert np.isclose(interpolate(np.array([1.0]), tensors), nodal_values[-1])
    with pytest.raises(ValueError, match="unit hypercube"):
        interpolate(np.array([1.5]), tensors)
    # the exact-1.0 remap must not swallow overshoots on the direct
    # subspace path either (which skips the drivers' domain validation)
    level, subspace = next(iter(tensors.subspaces.items()))
    with pytest.raises(IndexError):
        interpolate_subspace(
            tensors.scaling_dimensions(level), np.array([[1.0 + 1e-9]]), subspace
        )

    nodal_2d = np.arange(16.0).reshape(4, 4)
    tensors_2d = hierarchize(nodal_2d)
    corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = interpolate(corners, tensors_2d)
    assert np.allclose(
        values, [nodal_2d[0, 0], nodal_2d[-1, 0], nodal_2d[0, -1], nodal_2d[-1, -1]]
    )


def test_interpolate_and_compress_handle_empty_subspaces():
    tensors = hierarchize(np.random.default_rng(22).random((4, 4)))
    emptied_level = (2, 2)
    tensors.subspaces[emptied_level] = replace(
        tensors.subspaces[emptied_level], data=None
    )

    # EMPTY subspaces contribute zero, matching dehierarchize's zero blocks
    midpoints = midpoint_coordinates_from_level(tensors.max_level)
    assert np.allclose(interpolate(midpoints, tensors), dehierarchize(tensors))

    # and compress treats them as all-zero (dropped here, since not lmin)
    compressed = compress(tensors, only_whole_subspaces=True)
    assert emptied_level not in compressed.subspaces
