# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.typing as npt
from typing import TYPE_CHECKING, Sequence


from spght.basis import CellCentered
from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.lifting import Basis1D, BasisLike, as_bases, evaluate_block
from spght.linearize import coordinates_to_multidim_indices
from spght.pywt_compat import is_pywt_wavelet, reconstruct_block_haar_pywt
from spght.wavelets import haar, half_haar

if TYPE_CHECKING:
    import pywt


def _is_cell_haar(basis: Basis1D) -> bool:
    return (
        isinstance(basis.centering, CellCentered) and basis.scheme.steps == haar.steps
    )


def _cell_indices_clipped(
    coordinates: npt.NDArray, extents: Sequence[int]
) -> npt.NDArray:
    """Cell indices for coordinates in [0, 1]^d; the upper domain boundary
    (coordinate exactly 1.0) belongs to the last cell. Coordinates beyond
    1.0 keep their out-of-range index and fail the downstream bounds
    checks, exactly like any other out-of-domain coordinate."""
    indices = coordinates_to_multidim_indices(coordinates, tuple(extents))
    last_cell = np.asarray(extents, dtype=np.int64) - 1
    return np.where(coordinates == 1.0, last_cell, indices)


def _evaluate_subspace_haar(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    subspace: Subspace,
) -> npt.NDArray:
    """Fast path: evaluate the subspace directly, without reconstruction."""
    assert subspace.data is not None
    doubled_extents = tuple(2 * extent for extent in subspace.extents)
    cell_indices = _cell_indices_clipped(coordinates, doubled_extents)
    coefficient_indices = cell_indices // 2
    signs = np.ones(coordinates.shape[0])
    for d, is_scaling in enumerate(scaling_dimensions):
        if not is_scaling:
            signs *= 1.0 - 2.0 * (cell_indices[:, d] % 2)  # even -> +1, odd -> -1
    return subspace.data[coefficient_indices] * signs


def _synthesize_block_haar(
    coefficients: npt.NDArray, scaling_dimensions: Sequence[bool]
) -> npt.NDArray:
    """One inverse Haar step per dimension, in plain numpy: a scaling
    coefficient duplicates into both children ([c, c]), a detail
    coefficient contributes with alternating sign ([c, -c]). Bit-identical
    to the classical filter-bank reconstruction in
    spght.pywt_compat.reconstruct_block_haar_pywt."""
    for d, is_scaling in enumerate(scaling_dimensions):
        # one inverse transform step along each dimension doubles its extent
        coefficients = np.repeat(coefficients, 2, axis=d)
        if not is_scaling:
            odd_cells = [slice(None)] * coefficients.ndim
            odd_cells[d] = slice(1, None, 2)
            coefficients[tuple(odd_cells)] *= -1
    return coefficients


def _reconstruct_subspace_and_evaluate(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    subspace: Subspace,
    use_pywt: bool = False,
) -> npt.NDArray:
    """Reconstruct the subspace's function on the once-refined grid with
    the inverse Haar transform, then evaluate by cell lookup. With
    `use_pywt`, the equivalent classical filter-bank reconstruction is used
    (requires the optional PyWavelets dependency)."""
    assert subspace.data is not None
    # the n-d view works for dense and sparse alike (dropped coefficients
    # read as zeros); the quantization fields are reserved and not yet applied
    coeffs = subspace.data.to_dense().astype(np.float64)
    if use_pywt:
        coeffs = reconstruct_block_haar_pywt(coeffs, scaling_dimensions)
    else:
        coeffs = _synthesize_block_haar(coeffs, scaling_dimensions)

    # evaluate by piecewise-constant cell lookup;
    # only valid for the Haar scaling function
    multidim_indices = _cell_indices_clipped(coordinates, coeffs.shape)
    return coeffs[tuple(multidim_indices.T)]


def interpolate_subspace(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    subspace: Subspace,
    wavelet: "pywt.Wavelet | BasisLike | None" = None,
) -> npt.NDArray:
    """Interpolate a single subspace's contribution at the given coordinates.

    `wavelet` is a Basis1D (or one per dimension); None means the
    cell-centered Haar basis. A pywt.Wavelet may also be passed to
    reconstruct with the classical filter-bank machinery instead (requires
    the optional PyWavelets dependency; only the half_haar filter bank is
    supported there)."""
    num_dims = len(subspace.extents)
    assert len(coordinates.shape) == 2 and coordinates.shape[1] == num_dims
    assert len(scaling_dimensions) == num_dims
    if subspace.data is None:
        # EMPTY subspace: every coefficient is implicitly zero
        return np.zeros(coordinates.shape[0])
    if wavelet is not None and is_pywt_wavelet(wavelet):
        if wavelet != half_haar:
            raise NotImplementedError(
                "Only the half_haar filter bank is supported for pywt-based "
                "interpolation"
            )
        # explicit pywt input: demonstrate the classical reconstruction
        return _reconstruct_subspace_and_evaluate(
            scaling_dimensions, coordinates, subspace, use_pywt=True
        )
    if wavelet is not None:
        bases = as_bases(wavelet, num_dims)  # type: ignore[arg-type]
        if not all(_is_cell_haar(basis) for basis in bases):
            # lifting basis: reconstruct detail dimensions one level, then
            # evaluate with the basis' own stencils (constant / linear)
            return evaluate_block(
                scaling_dimensions, coordinates, subspace.data.to_dense(), bases
            )
    # cell-centered Haar: evaluate directly unless a large batch amortizes
    # the full reconstruction
    if subspace.data.is_sparse or coordinates.shape[0] < subspace.data.size:
        return _evaluate_subspace_haar(scaling_dimensions, coordinates, subspace)
    return _reconstruct_subspace_and_evaluate(scaling_dimensions, coordinates, subspace)


def interpolate_single_coordinate(
    coordinate: Sequence[float] | npt.NDArray,
    spghtensors: SparseGridHierarchicalTensors,
) -> float:
    """Interpolate a single coordinate in [0, 1]^d using the sparse grid
    hierarchical tensors"""
    wavelet = spghtensors.bases
    coordinate_np = np.asarray(coordinate)
    coordinate_np_two_d = coordinate_np.reshape(1, -1)
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    if not (np.all(coordinate_np >= 0.0) and np.all(coordinate_np <= 1.0)):
        raise ValueError("Coordinates must be within the unit hypercube [0, 1]^d")
    # iterate over the subspaces in the SparseGridHierarchicalTensors
    # and interpolate on each of them
    value: float = 0.0
    for level, subspace in spghtensors.subspaces.items():
        value += interpolate_subspace(
            spghtensors.scaling_dimensions(level),
            coordinate_np_two_d,
            subspace,
            wavelet=wavelet,
        )[0]

    return float(value)


def interpolate_many_coordinates(
    coordinates: Sequence[Sequence[float]] | npt.NDArray,
    spghtensors: SparseGridHierarchicalTensors,
) -> np.ndarray:
    """Interpolate many coordinates in [0, 1]^d using the sparse grid hierarchical tensors."""
    wavelet = spghtensors.bases
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    coordinates_np = np.asarray(coordinates)
    if not (np.all(coordinates_np >= 0.0) and np.all(coordinates_np <= 1.0)):
        raise ValueError("Coordinates must be within the unit hypercube [0, 1]^d")
    *batch_shape, num_dims = coordinates_np.shape

    flat_coords = coordinates_np.reshape(-1, num_dims)

    # Accumulate subspace contributions in double precision!
    flat_values = np.zeros(flat_coords.shape[:-1], dtype=np.float64)
    for level, subspace in spghtensors.subspaces.items():
        flat_values += interpolate_subspace(
            spghtensors.scaling_dimensions(level),
            flat_coords,
            subspace,
            wavelet=wavelet,
        )

    return np.asarray(flat_values).reshape(batch_shape)


def interpolate(
    coordinates: Sequence[float] | npt.NDArray,
    spghtensors: SparseGridHierarchicalTensors,
) -> float | npt.NDArray[np.float64]:
    """Evaluate the hierarchical tensors at coordinates in [0, 1]^d, using
    the basis recorded on the container.

    A single coordinate (a 1-D sequence of d numbers, of any numeric type)
    returns a scalar; a batch of shape (..., d) returns an array of the
    batch shape. Coordinates on 1.0 belong to the last cell."""
    coordinates_np = np.asarray(coordinates, dtype=np.float64)
    if coordinates_np.ndim == 0:
        raise ValueError(
            "Coordinates must be a sequence of d numbers or a batch of shape (..., d)"
        )
    if coordinates_np.ndim == 1:
        return interpolate_single_coordinate(coordinates_np, spghtensors)
    return interpolate_many_coordinates(coordinates_np, spghtensors)
