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
    assert np.isclose(value, 4.0)
