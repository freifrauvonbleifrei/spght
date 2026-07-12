import numpy as np

from spght.linearize import (
    coordinates_to_multidim_indices,
    indices_to_multidim_indices,
    multidim_indices_to_indices,
)


def test_coordinates_to_multidim_indices():
    extents = (2, 4, 8)
    coordinates = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.7, 0.7, 0.7]])
    multidim_indices = coordinates_to_multidim_indices(coordinates, extents)
    assert np.all(multidim_indices == [[0, 0, 0], [0, 0, 1], [1, 2, 5]])
    # at interval boundaries, assign to higher index
    # (in line with binary number intervals / location codes)
    coordinates = np.array([[0.5, 0.25, 0.75]])
    multidim_indices = coordinates_to_multidim_indices(coordinates, extents)
    assert np.all(multidim_indices == [[1, 1, 6]])


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


def test_multidim_indices_to_indices_and_back_multiple():
    extents = (2, 4, 3)
    for order in ["C", "F", "ZC", "ZF"]:
        multi_indices = np.array([[0, 1, 2], [1, 3, 0], [1, 2, 1]])
        indices = multidim_indices_to_indices(multi_indices, extents, order=order)
        recovered_multi_indices = indices_to_multidim_indices(
            indices, extents, order=order
        )
        assert np.all(recovered_multi_indices == multi_indices)
