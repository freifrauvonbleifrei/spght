# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Bridge to PyWavelets, showing how spght's transforms map onto the
classical filter-bank machinery.

spght's transforms use *half* normalization instead of the orthonormal
1/sqrt(2), because in the nested (sparse grid) view a higher level means
smaller basis-function support: coefficients are then directly the local
function values/surpluses. Expressed as a pywt filter bank
(dec_lo, dec_hi, rec_lo, rec_hi), the half-normalized Haar wavelet is
`half_haar_filters`; the lifting scheme `spght.wavelets.haar` computes the
bit-identical transform."""

from collections.abc import Sequence
import numpy.typing as npt
from typing import TYPE_CHECKING

from spght.util import depends_on_optional, module_is_available

if TYPE_CHECKING:
    import pywt

PYWT_AVAILABLE = module_is_available("pywt")

half_haar_filters = ([0.5, 0.5], [-0.5, 0.5], [1.0, 1.0], [1.0, -1.0])

#: the half-normalized Haar wavelet as a pywt object (None without pywt)
half_haar: "pywt.Wavelet | None"
if PYWT_AVAILABLE:
    import pywt

    half_haar = pywt.Wavelet(name="half_haar", filter_bank=half_haar_filters)
else:
    half_haar = None


def is_pywt_wavelet(wavelet: object) -> bool:
    """Whether `wavelet` is a pywt.Wavelet. Needs no import when pywt is
    missing: no pywt.Wavelet instance can exist then."""
    if not PYWT_AVAILABLE:
        return False
    import pywt

    return isinstance(wavelet, pywt.Wavelet)


@depends_on_optional("pywt")
def reconstruct_block_haar_pywt(
    coefficients: npt.NDArray, scaling_dimensions: Sequence[bool]
) -> npt.NDArray:
    """One inverse filter-bank step per dimension with the half_haar
    filters: scaling dimensions reconstruct from the approximation channel,
    detail dimensions from the detail channel. Bit-identical to the numpy
    synthesis in spght.interpolate; kept as the executable connection to
    the classical wavelet machinery."""
    import pywt

    # alternatively implemented in pywt directly as fswavedecn
    # https://github.com/PyWavelets/pywt/blob/1.6.x/doc/source/ref/2d-decompositions-overview.rst#fully-separable-discrete-wavelet-transform
    for d, is_scaling in enumerate(scaling_dimensions):
        if is_scaling:
            coefficients = pywt.idwt(coefficients, None, half_haar, axis=d)
        else:
            coefficients = pywt.idwt(None, coefficients, half_haar, axis=d)
    return coefficients
