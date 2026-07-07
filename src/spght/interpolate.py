import numpy as np
import pywt
from typing import Sequence

from icecream import ic

from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.linearization import coordinate_to_multidim_index, level_from_extent


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
    level: Sequence[int], coordinate: np.ndarray, subspace: Subspace, wavelet="haar"
) -> float:
    # interpolate on a single subspace using the wavelet transform
    # TODO reconstruct using only necessary coefficients!
    num_dims = len(subspace.extents)
    assert (
        len(coordinate) == num_dims
    ), "Coordinates dimensionality does not match subspace"
    # level = [level_from_extent(extent) for extent in subspace.extents]
    coeffs = subspace.values.copy()
    if sum(level) > 0:
        for d in range(num_dims):
            # for each 1-d pole in coeffs, we obtain a twice-as-long 1d array
            coeffs_detail_reconstructed = np.zeros(
                list(coeffs.shape[:d])
                + [coeffs.shape[d] * 2]
                + list(coeffs.shape[d + 1 :])
            )
            for idx in iter_pole_slices(coeffs.shape, axis=d):
                ic(
                    coeffs[idx],
                    coeffs[idx].shape,
                    coeffs_detail_reconstructed[idx].shape,
                    level[d],
                )
                coeffs_detail_reconstructed[idx] = ic(
                    pywt.upcoef(
                        "d",
                        coeffs[idx],
                        wavelet=wavelet,
                        level=1,
                    )
                )
            coeffs = coeffs_detail_reconstructed
        ic(coeffs_detail_reconstructed, coeffs_detail_reconstructed.shape, level)
    # evaluate scaling function at the given coordinates
    # phi, psi, x = pywt.Wavelet(wavelet).wavefun(level=1) + needs normalization for phi!
    assert (
        wavelet == "haar"
    ), "Only Haar wavelet is currently supported for interpolation"
    ic(coeffs, coeffs.shape, subspace.values)

    # interpolate using the scaling function at given coordinate
    value = coeffs[coordinate_to_multidim_index(coordinate, coeffs.shape)] / 2**((num_dims - 1) / 2)

    return ic(value)


def interpolate(
    coordinates: np.ndarray, spghtensors: SparseGridHierarchicalTensors, wavelet="haar"
) -> float:
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    if not np.all((coordinates >= 0) & (coordinates <= 1)):
        raise ValueError("Coordinates must be within the unit hypercube [0, 1]^d")
    # iterate over the subspaces in the SparseGridHierarchicalTensors
    # and interpolate on each of them
    value: float = 0.0
    for level, subspace in spghtensors.subspaces.items():
        ic(level)
        value += ic(interpolate_subspace(level, coordinates, subspace, wavelet=wavelet))
        ic(subspace)

    return value
