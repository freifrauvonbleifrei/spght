import numpy as np
import numpy.typing as npt
import pywt
from typing import Sequence


from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.linearize import coordinates_to_multidim_indices
from spght.wavelets import half_haar


def iter_pole_slices(shape, axis):
    """Yield index tuples selecting each 1D pole along `axis`."""
    ndim = len(shape)
    other_axes = [d for d in range(ndim) if d != axis]
    other_shape = tuple(shape[d] for d in other_axes)
    for idx in np.ndindex(other_shape):
        full_idx = [slice(None)] * ndim
        for ax, i in zip(other_axes, idx):
            full_idx[ax] = i
        yield tuple(full_idx)


def interpolate_subspace(
    level: Sequence[int],
    coordinates: npt.NDArray,
    subspace: Subspace,
    wavelet=half_haar,
) -> npt.NDArray:
    # interpolate on a single subspace using the wavelet transform
    # TODO reconstruct using only necessary coefficients
    num_dims = len(subspace.extents)
    assert len(coordinates.shape) == 2 and coordinates.shape[1] == num_dims
    assert subspace.data is not None
    # the n-d view works for dense and sparse alike (dropped coefficients
    # read as zeros); the quantization fields are reserved and not yet applied
    coeffs = subspace.data.to_dense()

    for d in range(num_dims):
        # for each 1-d pole in coeffs, we obtain a twice-as-long 1d array
        coeffs_detail_reconstructed = np.zeros(
            list(coeffs.shape[:d]) + [coeffs.shape[d] * 2] + list(coeffs.shape[d + 1 :])
        )
        # TODO "scaling-ness" / lmin-ness as separate parameter
        if level[d] == 0:
            mode = "a"
        else:
            mode = "d"
        for idx in iter_pole_slices(coeffs.shape, axis=d):
            coeffs_detail_reconstructed[idx] = pywt.upcoef(
                part=mode,
                coeffs=coeffs[idx],
                wavelet=wavelet,
                level=1,
            )
        coeffs = coeffs_detail_reconstructed

    # evaluate scaling function at the given coordinates
    # phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=1) + needs normalization for phi!
    assert (
        wavelet == half_haar
    ), "Only Haar wavelet is currently supported for interpolation"

    # interpolate using the scaling function at given coordinate
    multidim_indices = coordinates_to_multidim_indices(coordinates, coeffs.shape)
    value = coeffs[tuple(multidim_indices.T)]
    return value


def interpolate_single_coordinate(
    coordinate: Sequence[float],
    spghtensors: SparseGridHierarchicalTensors,
    wavelet=half_haar,
) -> float:
    """Interpolate a single coordinate in [0, 1]^d using the sparse grid hierarchical tensors."""
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
            level, coordinate_np_two_d, subspace, wavelet=wavelet
        )[0]

    return value


def interpolate_many_coordinates(
    coordinates: Sequence[Sequence[float]] | npt.NDArray,
    spghtensors: SparseGridHierarchicalTensors,
    wavelet=half_haar,
) -> np.ndarray:
    """Interpolate many coordinates in [0, 1]^d using the sparse grid hierarchical tensors."""
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
            level, flat_coords, subspace, wavelet=wavelet
        )

    return np.asarray(flat_values).reshape(batch_shape)


def interpolate(
    coordinates: Sequence[float],
    spghtensors: SparseGridHierarchicalTensors,
    wavelet=half_haar,
) -> float:
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    if isinstance(coordinates[0], float) or isinstance(coordinates[0], np.float32):
        return interpolate_single_coordinate(coordinates, spghtensors, wavelet=wavelet)
    elif isinstance(coordinates[0], (Sequence, np.ndarray)):
        return interpolate_many_coordinates(coordinates, spghtensors, wavelet=wavelet)
    else:
        raise ValueError("Unexpected type for coordinates")
