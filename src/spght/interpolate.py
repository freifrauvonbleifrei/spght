import numpy as np
import pywt
from typing import Sequence


from spght.data_structures import SparseGridHierarchicalTensors, Subspace
from spght.linearization import coordinate_to_multidim_index
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
    coordinate: Sequence[float],
    subspace: Subspace,
    wavelet=half_haar,
) -> float:
    # interpolate on a single subspace using the wavelet transform
    # TODO reconstruct using only necessary coefficients
    # TODO overload for many coordinates at once
    num_dims = len(subspace.extents)
    assert (
        len(coordinate) == num_dims
    ), "Coordinates dimensionality does not match subspace"
    coeffs = subspace.values.copy()  # type: ignore

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
    value = coeffs[coordinate_to_multidim_index(tuple(coordinate), coeffs.shape)]
    assert isinstance(value, float)
    return value


def interpolate(
    coordinates: Sequence[float],
    spghtensors: SparseGridHierarchicalTensors,
    wavelet=half_haar,
) -> float:
    # assert that all coordinates are within the unit hypercube [0, 1]^d
    if not np.all((coordinates >= 0) & (coordinates <= 1)):
        raise ValueError("Coordinates must be within the unit hypercube [0, 1]^d")
    # iterate over the subspaces in the SparseGridHierarchicalTensors
    # and interpolate on each of them
    value: float = 0.0
    for level, subspace in spghtensors.subspaces.items():
        value += interpolate_subspace(level, coordinates, subspace, wavelet=wavelet)

    return value
