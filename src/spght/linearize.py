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


Order = Literal["C", "F", "ZC", "ZF"]

MultiIndices = Union[Sequence[Sequence[int]], npt.NDArray[np.integer]]


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
    """Convert multiple linear indices to multi-dimensional indices."""
    num_dims = len(extents)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = idx.shape[0]

    multidim = np.zeros((n, num_dims), dtype=np.int64)

    if order in ("C", "F"):
        strides = _compute_strides(extents, order)
        for i in range(num_dims):
            multidim[:, i] = (idx // strides[i]) % extents[i]
    else:
        if order == "ZC":
            masks = _build_masks_zc(tuple(extents))
        elif order == "ZF":
            masks = _build_masks_zf(tuple(extents))
        else:
            raise ValueError(f"Unsupported order: {order}")
        multidim = _decode(idx, masks)

    return multidim


def multidim_indices_to_indices(
    multidim_indices: MultiIndices,
    extents: Sequence[int],
    order: Order,
) -> npt.NDArray[np.int64]:
    """Convert multiple multi-dimensional indices to linear indices."""
    num_dims = len(extents)

    idx = np.asarray(multidim_indices, dtype=np.int64)
    if idx.ndim == 1:
        idx = idx.reshape(-1, num_dims)
    elif idx.ndim != 2:
        raise ValueError(f"Expected a 1-D or 2-D array of indices, got ndim={idx.ndim}")
    if idx.shape[1] != num_dims:
        raise ValueError(
            f"Expected indices of dimensionality {num_dims}, got {idx.shape[1]}"
        )

    if order in ("C", "F"):
        strides = _compute_strides(extents, order)
        indices = np.zeros(idx.shape[0], dtype=np.int64)
        for i in range(num_dims):
            indices += idx[:, i] * strides[i]
    else:
        if order == "ZC":
            masks = _build_masks_zc(tuple(extents))
        elif order == "ZF":
            masks = _build_masks_zf(tuple(extents))
        else:
            raise ValueError(f"Unsupported order: {order}")
        indices = _encode(idx, masks)

    return indices


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
    return multidim_indices_to_indices(
        [coordinate_to_multidim_index(coordinates, extents)], extents, order
    )
