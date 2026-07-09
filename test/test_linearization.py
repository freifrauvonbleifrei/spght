import numpy as np

from spght.linearization import (
    coordinate_to_multidim_index,
)


def test_coordinate_to_multidim_index():
    extents = (2, 4, 8)
    coordinate = np.array([0.1, 0.1, 0.1])
    assert coordinate_to_multidim_index(coordinate, extents) == (0, 0, 0)
    coordinate = np.array([0.2, 0.2, 0.2])
    assert coordinate_to_multidim_index(coordinate, extents) == (0, 0, 1)
    coordinate = np.array([0.7, 0.7, 0.7])
    assert coordinate_to_multidim_index(coordinate, extents) == (1, 2, 5)
    # at interval boundaries, assign to higher index
    # (in line with binary number intervals / location codes)
    coordinate = np.array([0.5, 0.25, 0.75])
    multidim_index = coordinate_to_multidim_index(coordinate, extents)
    assert multidim_index == (1, 1, 6)
