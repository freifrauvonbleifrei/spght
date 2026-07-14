# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import numpy.typing as npt
import pywt
from typing import Sequence


from spght.basis import CellCentered
from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.lifting import Basis1D, as_bases, evaluate_block
from spght.linearize import coordinates_to_multidim_indices
from spght.wavelets import haar, half_haar


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


def _reconstruct_subspace_and_evaluate(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    subspace: Subspace,
    wavelet=half_haar,
) -> npt.NDArray:
    """Reference path: reconstruct the subspace's function on the
    once-refined grid with the inverse wavelet transform, then evaluate by
    cell lookup."""
    assert subspace.data is not None
    # the n-d view works for dense and sparse alike (dropped coefficients
    # read as zeros); the quantization fields are reserved and not yet applied
    coeffs = subspace.data.to_dense()

    for d, is_scaling in enumerate(scaling_dimensions):
        # one inverse transform step along each dimension doubles its extent
        if is_scaling:
            coeffs = pywt.idwt(coeffs, None, wavelet, axis=d)
        else:
            coeffs = pywt.idwt(None, coeffs, wavelet, axis=d)

    # evaluate by piecewise-constant cell lookup;
    # only valid for the Haar scaling function
    # phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=1) + needs normalization for phi!
    assert (
        wavelet == half_haar
    ), "Only Haar wavelet is currently supported for interpolation"
    multidim_indices = _cell_indices_clipped(coordinates, coeffs.shape)
    return coeffs[tuple(multidim_indices.T)]


def interpolate_subspace(
    scaling_dimensions: Sequence[bool],
    coordinates: npt.NDArray,
    subspace: Subspace,
    wavelet=half_haar,
) -> npt.NDArray:
    """Interpolate a single subspace's contribution at the given coordinates
    using wavelet transform."""
    num_dims = len(subspace.extents)
    assert len(coordinates.shape) == 2 and coordinates.shape[1] == num_dims
    assert len(scaling_dimensions) == num_dims
    if subspace.data is None:
        # EMPTY subspace: every coefficient is implicitly zero
        return np.zeros(coordinates.shape[0])
    if isinstance(wavelet, pywt.Wavelet):
        is_haar = wavelet == half_haar
    else:
        bases = as_bases(wavelet, num_dims)
        is_haar = all(_is_cell_haar(basis) for basis in bases)
        if not is_haar:
            # lifting basis: reconstruct detail dimensions one level, then
            # evaluate with the basis' own stencils (constant / linear)
            return evaluate_block(
                scaling_dimensions, coordinates, subspace.data.to_dense(), bases
            )
        wavelet = half_haar  # bit-identical, and unlocks the fast paths
    if is_haar and (
        subspace.data.is_sparse or coordinates.shape[0] < subspace.data.size
    ):
        return _evaluate_subspace_haar(scaling_dimensions, coordinates, subspace)
    return _reconstruct_subspace_and_evaluate(
        scaling_dimensions, coordinates, subspace, wavelet
    )


def interpolate_single_coordinate(
    coordinate: Sequence[float],
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

    return value


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
    coordinates: Sequence[float],
    spghtensors: SparseGridHierarchicalTensors,
) -> float:
    """Evaluate the hierarchical tensors at coordinates in [0, 1]^d, using
    the basis recorded on the container."""
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    if isinstance(coordinates[0], float) or isinstance(coordinates[0], np.float32):
        return interpolate_single_coordinate(coordinates, spghtensors)
    elif isinstance(coordinates[0], (Sequence, np.ndarray)):
        return interpolate_many_coordinates(coordinates, spghtensors)
    else:
        raise ValueError("Unexpected type for coordinates")
