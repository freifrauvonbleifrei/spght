# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import bitarray as ba
import bitarray.util as bau
from functools import lru_cache
import numpy as np
import numpy.typing as npt
import math
from typing import Literal, Sequence, Union


def level_from_extent(extent: int) -> int:
    # TODO may need more parameters, because what's needed
    # differs between scaling and hierarchical indexing (+1)
    return math.ceil(np.log2(extent))


def extent_from_level(level: int) -> int:
    return 2**level


Order = Literal["C", "F", "ZC", "ZF"]

MultiIndices = Union[Sequence[Sequence[int]], npt.NDArray[np.integer]]
IndexLike = Union[int, np.integer, Sequence[int], MultiIndices]
Coordinates = Union[Sequence[Sequence[float]], npt.NDArray[np.floating]]


def _all_powers_of_two(extents: Sequence[int]) -> bool:
    return all(e >= 1 and (e & (e - 1)) == 0 for e in extents)


def _check_multidim_bounds(idx: npt.NDArray[np.int64], extents: Sequence[int]) -> None:
    if np.any(idx < 0) or np.any(idx >= np.asarray(extents, dtype=np.int64)):
        raise IndexError(f"Multidim index out of bounds for extents {tuple(extents)}")


def _check_linear_bounds(linear: npt.NDArray[np.int64], extents: Sequence[int]) -> None:
    if np.any(linear < 0) or np.any(linear >= int(np.prod(extents))):
        raise IndexError(
            f"Linear index out of bounds for extents {tuple(extents)} "
            f"(size {int(np.prod(extents))})"
        )


## Z-order curves are similar to ALTO linearization
@lru_cache
def _build_masks(extent, order):
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
def _build_masks_zc(dims):
    return _build_masks(dims, tuple(range(len(dims))))


@lru_cache
def _build_masks_zf(dims):
    return _build_masks(dims, tuple(reversed(range(len(dims)))))


def _pdep(src: npt.NDArray[np.uint64], mask: int) -> npt.NDArray[np.uint64]:
    """Vectorized PDEP: scatter src's bits (LSB-first) into mask's set positions"""
    result = np.zeros_like(src)
    bb = 1
    m = mask
    while m:
        lsb = m & (-m)
        result |= np.where((src & bb) != 0, np.uint64(lsb), np.uint64(0))
        m &= m - 1
        bb <<= 1
    return result


def _pext(src: npt.NDArray[np.uint64], mask: int) -> npt.NDArray[np.uint64]:
    """Vectorized PEXT: gather the bits of src at mask's set positions, packed low."""
    result = np.zeros_like(src)
    bb = 1
    m = mask
    while m:
        lsb = m & (-m)
        result |= np.where((src & lsb) != 0, np.uint64(bb), np.uint64(0))
        m &= m - 1
        bb <<= 1
    return result


def _encode(
    indices: npt.NDArray[np.uint64], masks: Sequence[int]
) -> npt.NDArray[np.uint64]:
    indices = np.asarray(indices, dtype=np.uint64)
    pos = np.zeros(indices.shape[0], dtype=np.uint64)
    for n, mask in enumerate(masks):
        pos |= _pdep(indices[:, n], mask)
    return pos


def _decode(
    pos: npt.NDArray[np.uint64], masks: Sequence[int]
) -> npt.NDArray[np.uint64]:
    """Returns one multidim index per row."""
    pos = np.asarray(pos, dtype=np.uint64)
    out = np.empty((pos.shape[0], len(masks)), dtype=np.uint64)
    for n, mask in enumerate(masks):
        out[:, n] = _pext(pos, mask)
    return out


def _round_robin(extents: Sequence[int], order: Order) -> tuple[int, ...]:
    if order == "ZC":
        return tuple(range(len(extents)))
    elif order == "ZF":
        return tuple(reversed(range(len(extents))))
    raise ValueError(f"Unsupported order: {order}")


def _bisect_encode(
    idx: npt.NDArray[np.int64],
    extents: Sequence[int],
    dim_rotation: tuple[int, ...],
) -> npt.NDArray[np.int64]:
    """Bijective Z-order for arbitrary extents: visit dimensions round-robin
    like _build_masks (a dimension drops out after ceil(log2(extent)) rounds),
    but split the dimension's current extent into ceil/floor halves instead of
    consuming a bit; a cell in the right half is preceded by all cells of the
    left half. For power-of-two extents this reproduces the mask-based order
    exactly, so those shapes take the faster _encode/_decode path.

    idx: (n, d) array of multidim indices."""
    idx = idx.copy()
    num_indices = idx.shape[0]
    current = np.broadcast_to(np.asarray(extents, dtype=np.int64), idx.shape).copy()
    total = np.full(num_indices, int(np.prod(extents)), dtype=np.int64)
    pos = np.zeros(num_indices, dtype=np.int64)
    for _ in range(max(level_from_extent(e) for e in extents)):
        for n in dim_rotation:
            col = current[:, n]
            splittable = col > 1
            if not splittable.any():
                continue
            left = (col + 1) >> 1  # ceil half
            rest = total // col  # cells per unit slab of dimension n
            go_right = splittable & (idx[:, n] >= left)
            pos += np.where(go_right, left * rest, 0)
            idx[:, n] -= np.where(go_right, left, 0)
            new_col = np.where(splittable, np.where(go_right, col - left, left), col)
            current[:, n] = new_col
            total = new_col * rest
    return pos


def _bisect_decode(
    pos: npt.NDArray[np.int64],
    extents: Sequence[int],
    dim_rotation: tuple[int, ...],
) -> npt.NDArray[np.int64]:
    """Inverse of _bisect_encode. Returns one multidim index per row."""
    pos = pos.copy()
    num_indices = pos.shape[0]
    num_dims = len(extents)
    current = np.broadcast_to(
        np.asarray(extents, dtype=np.int64), (num_indices, num_dims)
    ).copy()
    total = np.full(num_indices, int(np.prod(extents)), dtype=np.int64)
    out = np.zeros((num_indices, num_dims), dtype=np.int64)
    for _ in range(max(level_from_extent(e) for e in extents)):
        for n in dim_rotation:
            col = current[:, n]
            splittable = col > 1
            if not splittable.any():
                continue
            left = (col + 1) >> 1
            rest = total // col
            left_size = left * rest
            go_right = splittable & (pos >= left_size)
            pos -= np.where(go_right, left_size, 0)
            out[:, n] += np.where(go_right, left, 0)
            new_col = np.where(splittable, np.where(go_right, col - left, left), col)
            current[:, n] = new_col
            total = new_col * rest
    return out


def _compute_strides(extents: Sequence[int], order: Order) -> list[int]:
    """Per-dimension linear-index strides for row-major ('C') or column-major ('F') order."""
    num_dims = len(extents)
    strides = [1] * num_dims
    if order == "C":
        for i in reversed(range(num_dims - 1)):
            strides[i] = strides[i + 1] * extents[i + 1]
    elif order == "F":
        for i in range(1, num_dims):
            strides[i] = strides[i - 1] * extents[i - 1]
    else:
        raise ValueError(f"_compute_strides only supports 'C'/'F', got: {order}")
    return strides


def indices_to_multidim_indices(
    indices: Sequence[int] | npt.NDArray,
    extents: tuple[int, ...],
    order: Order,
) -> npt.NDArray[np.int64]:
    """Convert multiple linear indices to multi-dimensional indices.
    Linear indices outside [0, prod(extents)) raise IndexError."""
    num_dims = len(extents)
    idx: npt.NDArray[np.int64] = np.array(indices, dtype=np.int64).reshape(-1)
    n = idx.shape[0]
    _check_linear_bounds(idx, extents)

    multidim: npt.NDArray[np.int64] = np.zeros(shape=(n, num_dims), dtype=np.int64)

    if order in ("C", "F"):
        strides = _compute_strides(extents, order)
        for i in range(num_dims):
            multidim[:, i] = (idx // strides[i]) % extents[i]
    elif order in ("ZC", "ZF"):
        if _all_powers_of_two(extents):
            if order == "ZC":
                masks = _build_masks_zc(tuple(extents))
            else:
                masks = _build_masks_zf(tuple(extents))
            multidim = _decode(idx, masks)  # type: ignore
        else:
            multidim = _bisect_decode(idx, extents, _round_robin(extents, order))
    else:
        raise ValueError(f"Unsupported order: {order}")

    return multidim


def reshape_to_nxd(array: npt.NDArray, num_dims: int) -> npt.NDArray:
    idx = np.array(array)
    if idx.ndim == 1:
        idx = idx.reshape(-1, num_dims)
    elif idx.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D array of indices, got ndim={idx.ndim}")
    if idx.shape[1] != num_dims:
        raise ValueError(
            f"Expected indices of dimensionality {num_dims}, got {idx.shape[1]}"
        )
    return idx


def multidim_indices_to_indices(
    multidim_indices: MultiIndices,
    extents: Sequence[int],
    order: Order,
) -> npt.NDArray[np.int64]:
    """Convert multiple multi-dimensional indices to linear indices.
    Indices outside [0, extents) raise IndexError."""
    num_dims = len(extents)

    idx = np.array(multidim_indices, dtype=np.int64)
    idx = reshape_to_nxd(idx, num_dims)
    _check_multidim_bounds(idx, extents)

    if order in ("C", "F"):
        strides = _compute_strides(extents, order)
        indices = np.zeros(idx.shape[0], dtype=np.int64)
        for i in range(num_dims):
            indices += idx[:, i] * strides[i]
    elif order in ("ZC", "ZF"):
        if _all_powers_of_two(extents):
            if order == "ZC":
                masks = _build_masks_zc(tuple(extents))
            else:
                masks = _build_masks_zf(tuple(extents))
            indices = _encode(idx, masks)  # type: ignore
        else:
            indices = _bisect_encode(idx, extents, _round_robin(extents, order))
    else:
        raise ValueError(f"Unsupported order: {order}")

    return indices


def coordinates_to_multidim_indices(
    coordinates: Coordinates,
    extents: Sequence[int],
) -> npt.NDArray[np.int64]:
    """Convert coordinates in [0, 1]^d to multi-dimensional indices,
    assuming the extents cover exactly cells in the unit hypercube."""
    coordinates = np.asarray(coordinates)
    coordinates = reshape_to_nxd(coordinates, len(extents))

    cell_widths = np.array([1.0 / e for e in extents])
    multidim_indices = np.floor_divide(coordinates, cell_widths).astype(np.int64)
    return multidim_indices


def coordinate_to_multidim_index(
    coordinate: Sequence[float],
    extents: Sequence[int],
) -> tuple[int, ...]:
    return tuple(coordinates_to_multidim_indices([coordinate], extents)[0])


def coordinates_to_indices(
    coordinates: Coordinates,
    extents: Sequence[int],
    order: Order,
) -> npt.NDArray[np.int64]:
    return multidim_indices_to_indices(
        coordinates_to_multidim_indices(coordinates, extents), extents, order
    )


def midpoint_coordinates_from_level(
    level: Sequence[int] | npt.NDArray[np.integer],
) -> npt.NDArray[np.float64]:
    """Cell-midpoint coordinates in [0, 1]^d for a full grid of the given level."""
    dimensionality = len(level)
    extents = np.array([extent_from_level(lvl) for lvl in level], dtype=np.int64)
    unit_voxel_size = np.ones((dimensionality,), dtype=np.float64) / extents
    stacked_indices = np.meshgrid(
        *[np.arange(extent) for extent in extents],
        indexing="ij",
    )
    midpoints = np.stack(
        (
            *[
                (stacked_indices[i] + 0.5) * unit_voxel_size[i]
                for i in range(dimensionality)
            ],
        ),
        axis=-1,
    )
    return midpoints
