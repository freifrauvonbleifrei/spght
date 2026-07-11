import numpy as np

from spght.linearization import (
    coordinate_to_multidim_index,
    index_to_multidim_index,
    multidim_index_to_index,
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


def test_index_to_multidim_index_and_back():
    extents = (2, 4, 3)
    for order in ["C", "F"]:
        # TODO "ZC", "ZF"
        index_five = 5
        multidim_index_five = index_to_multidim_index(index_five, extents, order=order)
        assert (
            multidim_index_to_index(multidim_index_five, extents, order=order)
            == index_five
        )
        if order == "C":
            assert multidim_index_five == (0, 1, 2)
        elif order == "F":
            assert multidim_index_five == (1, 2, 0)

        index_fifteen = 15
        multidim_index_fifteen = index_to_multidim_index(
            index_fifteen, extents, order=order
        )
        assert (
            multidim_index_to_index(multidim_index_fifteen, extents, order=order)
            == index_fifteen
        )
        if order == "C":
            assert multidim_index_fifteen == (1, 1, 0)
        elif order == "F":
            assert multidim_index_fifteen == (1, 3, 1)


def test_all_index_to_multidim_index_and_back():
    extents = (1, 4, 8, 2)
    for order in ["C", "F", "ZC", "ZF"]:
        for index in range(np.prod(extents)):
            multidim_index = index_to_multidim_index(index, extents, order=order)
            print(order, index, multidim_index)
            assert (
                multidim_index_to_index(multidim_index, extents, order=order) == index
            )
