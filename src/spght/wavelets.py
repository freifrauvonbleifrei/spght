# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import pywt

from spght.basis import (
    BoundaryRule,
    CellCentered,
    Dirichlet,
    Extrapolate,
    Periodic,
    VertexCentered,
)
from spght.lifting import Basis1D, LiftingScheme, LiftingStep


def _periodic_vertex_basis(scheme: LiftingScheme, bc_left, bc_right) -> Basis1D:
    if bc_left is not None or bc_right is not None:
        raise ValueError("a periodic basis wraps on both sides; do not pass rules")
    return Basis1D(VertexCentered(periodic=True), scheme, Periodic(), Periodic())


# we want wavelets where we implicitly assume that higher level
# means smaller intervals (nesting)
# -> half normalization instead of 1/sqrt(2) normalization

half_haar_filters = ([0.5, 0.5], [-0.5, 0.5], [1.0, 1.0], [1.0, -1.0])

half_haar = pywt.Wavelet(name="half_haar", filter_bank=half_haar_filters)


# equivalent to half_haar:
haar = LiftingScheme(
    "haar",
    (
        LiftingStep("predict", (0,), (1.0,)),
        LiftingStep("scale_detail", factor=-0.5),
        LiftingStep("update", (0,), (-1.0,)),
    ),
    centering_kind="cell",
    evaluation="midpoint",
)

# the hierarchical hat basis == linear interpolet == lazy wavelet
hierarchical_hat = LiftingScheme(
    "hierarchical_hat",
    (LiftingStep("predict", (0, 1), (0.5, 0.5)),),
    centering_kind="vertex",
    evaluation="nodal_linear",
)

# linear predict plus the mean-preserving update (CDF(2,2)-style)
cdf_2_2 = LiftingScheme(
    "cdf_2_2",
    (
        LiftingStep("predict", (0, 1), (0.5, 0.5)),
        LiftingStep("update", (-1, 0), (0.25, 0.25)),
    ),
    centering_kind="vertex",
    evaluation="nodal_linear",
)

# cubic (Deslauriers-Dubuc order 4) interpolet
cubic_interpolet = LiftingScheme(
    "cubic_interpolet",
    (LiftingStep("predict", (-1, 0, 1, 2), (-1 / 16, 9 / 16, 9 / 16, -1 / 16)),),
    centering_kind="vertex",
    evaluation=None,
)


def haar_basis() -> Basis1D:
    """Cell-centered Haar; bit-identical to the pywt half_haar path."""
    return Basis1D(CellCentered(), haar)


def hat_basis(
    bc_left: BoundaryRule | None = None,
    bc_right: BoundaryRule | None = None,
    include_boundary: bool = True,
    periodic: bool = False,
) -> Basis1D:
    """The hierarchical hat basis; details are sparse grid surpluses."""
    if periodic:
        return _periodic_vertex_basis(hierarchical_hat, bc_left, bc_right)
    if not include_boundary:
        bc_left = bc_left if bc_left is not None else Dirichlet(0.0)
        bc_right = bc_right if bc_right is not None else Dirichlet(0.0)
    return Basis1D(
        VertexCentered(include_boundary),
        hierarchical_hat,
        bc_left if bc_left is not None else Extrapolate(),
        bc_right if bc_right is not None else Extrapolate(),
    )


def cdf_2_2_basis(
    bc_left: BoundaryRule | None = None,
    bc_right: BoundaryRule | None = None,
    periodic: bool = False,
) -> Basis1D:
    if periodic:
        return _periodic_vertex_basis(cdf_2_2, bc_left, bc_right)
    return Basis1D(
        VertexCentered(include_boundary=True),
        cdf_2_2,
        bc_left if bc_left is not None else Extrapolate(),
        bc_right if bc_right is not None else Extrapolate(),
    )


def cubic_basis(
    bc_left: BoundaryRule | None = None,
    bc_right: BoundaryRule | None = None,
    periodic: bool = False,
) -> Basis1D:
    if periodic:
        return _periodic_vertex_basis(cubic_interpolet, bc_left, bc_right)
    return Basis1D(
        VertexCentered(include_boundary=True),
        cubic_interpolet,
        bc_left if bc_left is not None else Extrapolate(),
        bc_right if bc_right is not None else Extrapolate(),
    )
