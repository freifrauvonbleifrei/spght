import numpy as np
import math
from typing import Literal, Sequence


## Z-order curves are analogous to ALTO linearization, modulo fitting to the extents exactly


def level_from_extent(extent: int) -> int:
    # TODO may need more parameters
    return math.ceil(np.log2(extent))


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
