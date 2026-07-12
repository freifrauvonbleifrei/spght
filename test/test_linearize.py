import numpy as np

from spght.linearize import (
    coordinate_to_multidim_index,
    indices_to_multidim_indices,
    multidim_indices_to_indices,
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


def test_indices_to_multidim_indices_and_back():
    extents = (2, 4, 3)
    for order in ["C", "F", "ZC", "ZF"]:
        index_five = 5
        multidim_index_five = indices_to_multidim_indices(
            [index_five], extents, order=order
        )[0]
        assert (
            multidim_indices_to_indices([multidim_index_five], extents, order=order)[0]
            == index_five
        )
        if order == "C":
            assert np.all(multidim_index_five == (0, 1, 2))
        elif order == "F":
            assert np.all(multidim_index_five == (1, 2, 0))
        elif order == "ZC":
            assert np.all(multidim_index_five == (0, 0, 3))
        elif order == "ZF":
            assert np.all(multidim_index_five == (1, 1, 0))

        index_fifteen = 15
        multidim_index_fifteen = indices_to_multidim_indices(
            [index_fifteen], extents, order=order
        )[0]
        assert (
            multidim_indices_to_indices([multidim_index_fifteen], extents, order=order)[
                0
            ]
            == index_fifteen
        )
        if order == "C":
            assert np.all(multidim_index_fifteen == (1, 1, 0))
        elif order == "F":
            assert np.all(multidim_index_fifteen == (1, 3, 1))
        elif order == "ZC":
            assert np.all(multidim_index_fifteen == (0, 3, 3))
        elif order == "ZF":
            assert np.all(multidim_index_fifteen == (1, 3, 1))


def test_all_indices_to_multidim_indices_and_back():
    extents = (1, 4, 8, 2)
    for order in ["C", "F", "ZC", "ZF"]:
        for index in range(np.prod(extents)):
            multidim_index = indices_to_multidim_indices([index], extents, order=order)[
                0
            ]
            assert (
                multidim_indices_to_indices([multidim_index], extents, order=order)[0]
                == index
            )
