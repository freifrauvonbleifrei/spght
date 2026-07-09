import bitarray as ba
import bitarray.util as bau
import numpy as np
import math
from typing import Literal, Sequence


## Z-order curves are analogous to ALTO linearization, modulo fitting to the extents exactly


def level_from_extent(extent: int) -> int:
    # TODO may need more parameters
    return math.ceil(np.log2(extent))


def index_to_multidim_index(
    index: int, extents: tuple[int, ...], order: Literal["C", "F", "ZC", "ZF"]
) -> tuple[int, ...]:
    """Convert a linear index to a multi-dimensional index."""
    multidim_index: list[int] = [0] * len(extents)
    if order == "C":
        accumulated_product = 1
        for i in reversed(range(len(extents))):
            multidim_index[i] = (index // accumulated_product) % extents[i]
            accumulated_product *= extents[i]
    elif order == "F":
        accumulated_product = 1
        for i in range(len(extents)):
            multidim_index[i] = (index // accumulated_product) % extents[i]
            accumulated_product *= extents[i]
    else:
        raise NotImplementedError("Z-order curves not implemented yet")
    return tuple(multidim_index)


def multidim_index_to_index(
    multidim_index: Sequence[int],
    extents: Sequence[int],
    order: Literal["C", "F", "ZC", "ZF"],
) -> int:
    """Convert a multi-dimensional index to a linear index."""
    index = 0
    if order == "C":
        for i in range(len(extents)):
            index = index * extents[i] + multidim_index[i]
    elif order == "F":
        for i in reversed(range(len(extents))):
            index = index * extents[i] + multidim_index[i]
    else:
        level = [level_from_extent(extent) + 1 for extent in extents]
        assert all(
            e == 2 ** (l - 1) for e, l in zip(extents, level)
        ), "Extents must be powers of two for Z-order curves."
        multidim_bit_index: list[ba.bitarray] = [
            bau.int2ba(idx, length=2**l) for idx, l in zip(multidim_index, level)
        ]
        raise NotImplementedError("Z-order curves not implemented yet")
    return index


def coordinate_to_multidim_index(
    coordinates: Sequence[float],
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
    order: Literal["C", "F", "ZC", "ZF"],
) -> int:
    return multidim_index_to_index(
        coordinate_to_multidim_index(coordinates, extents), extents, order
    )
