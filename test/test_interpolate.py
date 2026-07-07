import numpy as np
from icecream import ic

from spght.interpolate import interpolate
from spght.hierarchize import hierarchize


def test_interpolate_1d_level2():
    for coordinate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        nodal_values = np.array([1.0, -1.0, 1.0, -1.0])
        hierarchical_tensors = hierarchize(nodal_values)
        ic(hierarchical_tensors.subspaces)
        value = interpolate(np.array([coordinate]), hierarchical_tensors)
        if coordinate < 0.25:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.5:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.75:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        else:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"


def test_interpolate_1d_level3():
    for coordinate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        nodal_values = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        hierarchical_tensors = hierarchize(nodal_values)
        ic(hierarchical_tensors.subspaces)
        value = interpolate(np.array([coordinate]), hierarchical_tensors)
        if coordinate < 0.125:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.25:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.375:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.5:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.625:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.75:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"
        elif coordinate < 0.875:
            assert np.isclose(
                value, 1.0
            ), f"Expected 1.0 at coordinate {coordinate}, got {value}"
        else:
            assert np.isclose(
                value, -1.0
            ), f"Expected -1.0 at coordinate {coordinate}, got {value}"
