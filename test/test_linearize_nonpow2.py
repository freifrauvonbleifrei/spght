# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from spght.linearize import (
    indices_to_multidim_indices,
    multidim_indices_to_indices,
)


NON_POW2_EXTENTS = [
    (3,),
    (5, 3),
    (2, 7),
    (2, 4, 3),
    (9, 5, 3),
    (6, 10, 15),
    # sparse-grid shapes, 2^l + 1 (often prime, so no small radices to exploit)
    (5, 5),
    (17, 17),
    (9, 17, 5),
]


def _all_multidim_indices(extents):
    grids = np.meshgrid(*[np.arange(e) for e in extents], indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


def test_powers_of_two_keep_mask_order():
    # pinned values from the mask-based (pdep/pext) linearization; the
    # bisection construction must coincide with it for power-of-two extents
    extents = (8, 4, 16)
    points = np.array([[1, 2, 3], [7, 3, 15], [0, 1, 0], [5, 0, 9]])
    assert np.array_equal(
        multidim_indices_to_indices(points, extents, "ZC"), [135, 511, 16, 325]
    )
    assert np.array_equal(
        multidim_indices_to_indices(points, extents, "ZF"), [135, 511, 16, 323]
    )
    for order in ["ZC", "ZF"]:
        all_idx = _all_multidim_indices(extents)
        pos = multidim_indices_to_indices(all_idx, extents, order)
        assert np.array_equal(indices_to_multidim_indices(pos, extents, order), all_idx)


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
@pytest.mark.parametrize("extents", NON_POW2_EXTENTS)
def test_bijection_onto_dense_range(extents, order):
    all_idx = _all_multidim_indices(extents)
    pos = multidim_indices_to_indices(all_idx, extents, order)
    assert np.array_equal(np.sort(pos), np.arange(np.prod(extents)))


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
@pytest.mark.parametrize("extents", NON_POW2_EXTENTS)
def test_roundtrip_both_ways(extents, order):
    all_idx = _all_multidim_indices(extents)
    pos = multidim_indices_to_indices(all_idx, extents, order)
    assert np.array_equal(indices_to_multidim_indices(pos, extents, order), all_idx)
    linear = np.arange(np.prod(extents))
    back = multidim_indices_to_indices(
        indices_to_multidim_indices(linear, extents, order), extents, order
    )
    assert np.array_equal(back, linear)


@pytest.mark.parametrize("order", ["ZC", "ZF"])
@pytest.mark.parametrize("extents", NON_POW2_EXTENTS)
def test_decoded_indices_stay_in_bounds(extents, order):
    # unlike the padded mask construction, decoding any dense linear index
    # must yield a valid multidim index
    linear = np.arange(np.prod(extents))
    multidim = indices_to_multidim_indices(linear, extents, order)
    assert np.all(multidim >= 0)
    assert np.all(multidim < np.asarray(extents))


@pytest.mark.parametrize("order", ["ZC", "ZF"])
@pytest.mark.parametrize("extents", NON_POW2_EXTENTS)
def test_corners(extents, order):
    origin = np.zeros((1, len(extents)), dtype=np.int64)
    assert multidim_indices_to_indices(origin, extents, order)[0] == 0
    far_corner = np.asarray(extents, dtype=np.int64).reshape(1, -1) - 1
    assert (
        multidim_indices_to_indices(far_corner, extents, order)[0]
        == np.prod(extents) - 1
    )


@pytest.mark.parametrize("order", ["ZC", "ZF"])
@pytest.mark.parametrize("extents", [(5, 3), (17, 17), (9, 5, 3), (6, 10, 15)])
def test_first_split_halves_are_contiguous(extents, order):
    # the first round-robin split is along the first dimension of the rotation
    # (ZC: dim 0, ZF: last dim); the ceil-half sub-box must come first as one
    # contiguous index range
    split_dim = 0 if order == "ZC" else len(extents) - 1
    all_idx = _all_multidim_indices(extents)
    pos = multidim_indices_to_indices(all_idx, extents, order)
    left = (extents[split_dim] + 1) // 2
    left_size = left * np.prod(extents) // extents[split_dim]
    in_left_half = all_idx[:, split_dim] < left
    assert np.all(pos[in_left_half] < left_size)
    assert np.all(pos[~in_left_half] >= left_size)


@pytest.mark.parametrize("order", ["ZC", "ZF"])
def test_dense_gather_scatter_roundtrip(order):
    # emulates DenseTensor.from_dense / to_dense for a non-power-of-two shape
    extents = (5, 9, 3)
    rng = np.random.default_rng(42)
    array = rng.random(extents)
    coords = indices_to_multidim_indices(np.arange(array.size), extents, order)
    flat = array[tuple(coords.T)]
    restored = np.empty(extents, dtype=array.dtype)
    restored[tuple(coords.T)] = flat
    assert np.array_equal(restored, array)


def test_interleaves_prime_extents():
    # 17 = 2^4 + 1 is prime: digit/radix approaches degenerate to C order
    # there, but ceil/floor bisection must still interleave. Compare the mean
    # linear-index jump between spatially adjacent cells against C order.
    extents = (17, 17)
    all_idx = _all_multidim_indices(extents)
    pos = multidim_indices_to_indices(all_idx, extents, "ZC").reshape(extents)
    jump_dim0 = np.abs(np.diff(pos, axis=0)).mean()
    assert jump_dim0 < 17  # C order: exactly 17
    jump_dim1 = np.abs(np.diff(pos, axis=1)).mean()
    assert jump_dim1 > 1  # C order: exactly 1; interleaving spreads dim 1


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
@pytest.mark.parametrize("extents", [(2, 4, 3), (8, 4), (17, 17)])
def test_out_of_bounds_multidim_indices_raise(extents, order):
    too_large = list(extents)
    too_large[-1] = extents[-1]  # one past the last valid index
    with pytest.raises(IndexError):
        multidim_indices_to_indices([too_large], extents, order)
    negative = [0] * len(extents)
    negative[0] = -1
    with pytest.raises(IndexError):
        multidim_indices_to_indices([negative], extents, order)
    # a single offender within a batch of valid indices must also raise
    valid = [0] * len(extents)
    with pytest.raises(IndexError):
        multidim_indices_to_indices([valid, too_large, valid], extents, order)


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
@pytest.mark.parametrize("extents", [(2, 4, 3), (8, 4), (17, 17)])
def test_out_of_bounds_linear_indices_raise(extents, order):
    size = int(np.prod(extents))
    with pytest.raises(IndexError):
        indices_to_multidim_indices([size], extents, order)
    with pytest.raises(IndexError):
        indices_to_multidim_indices([-1], extents, order)
    with pytest.raises(IndexError):
        indices_to_multidim_indices([0, size, size - 1], extents, order)


@pytest.mark.parametrize("extents", [(4, 8), (5, 3, 7)])
def test_zc_zf_reverse_dimension_rotation(extents):
    all_idx = _all_multidim_indices(extents)
    zc = multidim_indices_to_indices(all_idx, extents, "ZC")
    zf_reversed = multidim_indices_to_indices(
        all_idx[:, ::-1], tuple(reversed(extents)), "ZF"
    )
    assert np.array_equal(zc, zf_reversed)
