import numpy as np

from spght.interpolate import interpolate
from spght.hierarchize import hierarchize


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
