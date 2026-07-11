import bitarray as ba
import bitarray.util as bau
from functools import lru_cache
import numpy as np
import math
from typing import Literal, Sequence


def level_from_extent(extent: int) -> int:
    # TODO may need more parameters, because what's needed
    # differs between scaling and hierarchical indexing (+1)
    return math.ceil(np.log2(extent))


Order = Literal["C", "F", "ZC", "ZF"]


## Z-order curves are similar to ALTO linearization
@lru_cache
def build_masks(extent, order):
    """
    order: round-robin visiting sequence of dimension indices used to
    assign successive bit positions. A dimension drops out of the
    rotation once it has been given all the bits it needs.
    """
    bits_needed = [level_from_extent(e) for e in extent]
    remaining = bits_needed[:]

    masks = [ba.bitarray(sum(bits_needed)) for _ in range(len(extent))]
    for m in masks:
        m.setall(0)

    pos = 0
    while any(r > 0 for r in remaining):
        for n in order:
            if remaining[n] > 0:
                masks[n][pos] = 1
                remaining[n] -= 1
                pos += 1
    int_masks = [bau.ba2int(m) for m in masks]
    return int_masks


@lru_cache
def build_masks_zc(dims):
    return build_masks(dims, tuple(range(len(dims))))


@lru_cache
def build_masks_zf(dims):
    return build_masks(dims, tuple(reversed(range(len(dims)))))


def pdep(src, mask):
    result = 0
    bb = 1
    while mask:
        lsb = mask & (-mask)
        if src & bb:
            result |= lsb
        mask &= mask - 1
        bb <<= 1
    return result


def pext(src, mask):
    result = 0
    bb = 1
    while mask:
        lsb = mask & (-mask)
        if src & lsb:
            result |= bb
        mask &= mask - 1
        bb <<= 1
    return result


def encode(index, masks):
    pos = 0
    for i_n, mask in zip(index, masks):
        pos |= pdep(i_n, mask)
    return pos


def decode(pos: int, masks: list[int]) -> list[int]:
    return [pext(pos, mask) for mask in masks]


def index_to_multidim_index(
    index: int, extents: tuple[int, ...], order: Order
) -> tuple[int, ...]:
    """Convert a linear index to a multi-dimensional index."""
    num_dims = len(extents)
    multidim_index: list[int] = [0] * num_dims
    if order == "C":
        accumulated_product = 1
        for i in reversed(range(num_dims)):
            multidim_index[i] = (index // accumulated_product) % extents[i]
            index //= extents[i]
    elif order == "F":
        accumulated_product = 1
        for i in range(num_dims):
            multidim_index[i] = (index // accumulated_product) % extents[i]
            accumulated_product *= extents[i]
    else:
        if order == "ZC":
            masks = build_masks_zc(tuple(extents))
        elif order == "ZF":
            masks = build_masks_zf(tuple(extents))
        else:
            raise ValueError(f"Unsupported order: {order}")

        multidim_index = decode(index, masks)
    return tuple(multidim_index)


def multidim_index_to_index(
    multidim_index: Sequence[int],
    extents: Sequence[int],
    order: Order,
) -> int:
    """Convert a multi-dimensional index to a linear index."""
    num_dims = len(extents)
    index = 0
    if order == "C":
        for i in range(num_dims):
            index = index * extents[i] + multidim_index[i]
    elif order == "F":
        for i in reversed(range(num_dims)):
            index = index * extents[i] + multidim_index[i]
    else:
        if order == "ZC":
            masks = build_masks_zc(tuple(extents))
        elif order == "ZF":
            masks = build_masks_zf(tuple(extents))
        else:
            raise ValueError(f"Unsupported order: {order}")
        index = encode(multidim_index, masks)
    return index


def coordinate_to_multidim_index(
    coordinates: Sequence[float] | np.ndarray,
    extents: Sequence[int],
) -> tuple[int, ...]:
    """Convert a coordinate in [0, 1]^d to a linear index,
    assuming the extents cover exactly cells in the unit hypercube."""
    assert len(coordinates) == len(
        extents
    ), "Coordinates dimensionality does not match extents"
    cell_widths = [1.0 / e for e in extents]
    multidim_index = tuple(
        int(np.floor(c / w)) for c, w in zip(coordinates, cell_widths)
    )
    return multidim_index


def coordinate_to_index(
    coordinates: Sequence[float],
    extents: Sequence[int],
    order: Order,
) -> int:
    return multidim_index_to_index(
        coordinate_to_multidim_index(coordinates, extents), extents, order
    )
