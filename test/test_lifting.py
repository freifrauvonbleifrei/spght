# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import numpy as np
import pytest
import pywt

from spght.basis import (
    CellCentered,
    Dirichlet,
    Extrapolate,
    Neumann,
    Periodic,
    VertexCentered,
)
from spght.compress import compress
from spght.hierarchize import dehierarchize, hierarchize
from spght.interpolate import interpolate
from spght.lifting import Basis1D
from spght.wavelets import (
    cdf_2_2_basis,
    cubic_basis,
    haar as haar_scheme,
    haar_basis,
    hat_basis,
    half_haar,
    hierarchical_hat,
)

RNG = np.random.default_rng(42)

ALL_RULES = [Periodic(), Neumann(), Dirichlet(0.7), Extrapolate()]


def _basis_id(basis):
    return f"{basis.centering.name}-{basis.scheme.name}-{type(basis.bc_left).__name__}"


def all_bases_1d():
    for bc in ALL_RULES:
        yield Basis1D(CellCentered(), haar_scheme, bc, bc), 16
        yield Basis1D(VertexCentered(True), hierarchical_hat, bc, bc), 17
        yield cdf_2_2_basis(bc, bc), 17
        yield cubic_basis(bc, bc), 17
    yield hat_basis(include_boundary=False), 15
    yield hat_basis(Dirichlet(1.0), Dirichlet(4.0), include_boundary=False), 15
    yield hat_basis(periodic=True), 16
    yield cdf_2_2_basis(periodic=True), 16
    yield cubic_basis(periodic=True), 16


ROUNDTRIP_BASES = [
    pytest.param(basis, extent, id=_basis_id(basis)) for basis, extent in all_bases_1d()
]


@pytest.mark.parametrize("min_level_offset", [0, 2], ids=["coarsest", "coarsest+2"])
@pytest.mark.parametrize("basis,extent", ROUNDTRIP_BASES)
def test_roundtrip_all_schemes_and_boundaries_1d(basis, extent, min_level_offset):
    nodal_values = RNG.normal(size=extent)
    min_level = basis.centering.lowest_min_level + min_level_offset
    hierarchical = hierarchize(nodal_values, wavelet=basis, min_level=min_level)
    back = dehierarchize(hierarchical, wavelet=basis)
    assert np.allclose(back, nodal_values, atol=1e-12)


@pytest.mark.parametrize(
    "shape,min_level",
    [((4, 8), 0), ((8, 8), (1, 2)), ((16,), 2), ((4, 4, 4), 1)],
    ids=["2d", "2d-min-level", "1d-min-level", "3d-min-level"],
)
def test_haar_basis_matches_pywt_half_haar(shape, min_level):
    # the lifting Haar must stay equivalent to the pywt half_haar transform

    nodal_values = RNG.normal(size=shape)
    via_lifting = hierarchize(nodal_values, wavelet=haar_basis(), min_level=min_level)

    num_dim = nodal_values.ndim
    minimum = [min_level] * num_dim if isinstance(min_level, int) else list(min_level)
    max_levels = [int(np.log2(extent)) for extent in shape]
    blocks = [nodal_values]
    for d in range(num_dim):
        num_levels = max_levels[d] - minimum[d]
        updated = []
        for block in blocks:
            if num_levels == 0:
                updated.append(block)
            else:
                updated.extend(pywt.wavedec(block, half_haar, axis=d, level=num_levels))
        blocks = updated
    labels = itertools.product(
        *(range(minimum[d], max_levels[d] + 1) for d in range(num_dim))
    )
    assert len(blocks) == len(via_lifting.subspaces)
    for label, block in zip(labels, blocks):
        assert np.allclose(
            via_lifting.subspaces[tuple(label)].data.to_dense(), block, atol=1e-12
        )


def test_hat_details_are_hierarchical_surpluses():
    x = np.linspace(0, 1, 17)
    hierarchical = hierarchize(3.0 * x + 1.0, wavelet=hat_basis())
    for level, subspace in hierarchical.subspaces.items():
        if level[0] > 0:
            assert np.allclose(subspace.data.to_dense(), 0.0, atol=1e-13)
    # and the known surplus of the standard parabola at the midpoint
    hierarchical = hierarchize(x * (1 - x), wavelet=hat_basis())
    assert np.isclose(hierarchical.subspaces[(1,)].data.to_dense()[0], 0.25)


@pytest.mark.parametrize(
    "g_left,g_right", [(0.0, 2.0), (1.0, 4.0)], ids=["half-homogeneous", "general"]
)
def test_interior_only_hat_with_dirichlet_data(g_left, g_right):
    x = np.linspace(0, 1, 17)
    basis = hat_basis(Dirichlet(g_left), Dirichlet(g_right), include_boundary=False)
    values = g_left + (g_right - g_left) * x
    hierarchical = hierarchize(values[1:-1], wavelet=basis, min_level=1)
    for level, subspace in hierarchical.subspaces.items():
        if level[0] > 1:
            assert np.allclose(subspace.data.to_dense(), 0.0, atol=1e-13)
    assert np.allclose(dehierarchize(hierarchical, wavelet=basis), values[1:-1])


def test_interior_only_grid_requires_min_level_one():
    basis = hat_basis(include_boundary=False)
    with pytest.raises(ValueError, match="min_level"):
        hierarchize(RNG.normal(size=15), wavelet=basis, min_level=0)


def test_cubic_interpolet_kills_cubics_in_the_interior():
    x = np.linspace(0, 1, 33)
    hierarchical = hierarchize(x**3, wavelet=cubic_basis())
    for level, subspace in hierarchical.subspaces.items():
        if level[0] > 1:
            # boundary stencils extrapolate linearly, -> only interior details vanish
            assert np.allclose(subspace.data.to_dense()[1:-1], 0.0, atol=1e-12)


# no Dirichlet case: for depth-1 ghosts, antisymmetric reflection about the
# wall dof coincides with linear extrapolation, and the cubic stencil never
# taps deeper on vertex grids -- so the rules are indistinguishable here
@pytest.mark.parametrize(
    "function,matched,mismatched",
    [
        pytest.param(
            lambda x: np.cos(np.pi * x), Neumann(), Extrapolate(), id="zero-slope"
        ),
        pytest.param(
            lambda x: np.sin(2 * np.pi * x), Periodic(), Neumann(), id="periodic"
        ),
    ],
)
def test_boundary_rules_shrink_matching_edge_details(function, matched, mismatched):
    x = np.linspace(0, 1, 33)

    def finest_edge_detail(hierarchical):
        finest = max(hierarchical.subspaces)
        values = hierarchical.subspaces[finest].data.to_dense()
        return max(abs(values[0]), abs(values[-1]))

    # ghosts of the matched rule are exact for the matched function, so its
    # edge details drop to the interior level while the mismatched rule's stay
    good = hierarchize(function(x), wavelet=cubic_basis(matched, matched))
    bad = hierarchize(function(x), wavelet=cubic_basis(mismatched, mismatched))
    assert finest_edge_detail(good) < 0.2 * finest_edge_detail(bad)


@pytest.mark.parametrize(
    "shape,bases,min_level",
    [
        pytest.param(
            (16, 17, 15),
            (
                haar_basis(),
                cdf_2_2_basis(Neumann(), Neumann()),
                hat_basis(include_boundary=False),
            ),
            (0, 0, 1),
            id="haar-cdf-hatinterior",
        ),
        pytest.param(
            (16, 17, 17),
            (
                Basis1D(CellCentered(), haar_scheme, Periodic(), Periodic()),
                cubic_basis(Neumann(), Neumann()),
                cdf_2_2_basis(Dirichlet(0.5), Dirichlet(0.5)),
            ),
            (1, 2, 0),
            id="periodic-cubic-dirichlet",
        ),
    ],
)
def test_anisotropic_mixed_roundtrip_3d(shape, bases, min_level):
    nodal_values = RNG.normal(size=shape)
    hierarchical = hierarchize(nodal_values, wavelet=bases, min_level=min_level)
    for level, subspace in hierarchical.subspaces.items():
        for d, basis in enumerate(bases):
            expected = (
                basis.centering.num_dofs(level[d])
                if level[d] == hierarchical.min_level[d]
                else basis.centering.num_details(level[d])
            )
            assert subspace.extents[d] == expected
    back = dehierarchize(hierarchical, wavelet=bases)
    assert np.allclose(back, nodal_values, atol=1e-14)


def test_dehierarchize_with_dropped_subspaces():
    x = np.linspace(0, 1, 17)
    nodal_values = np.sin(np.pi * x)[:, np.newaxis] * np.sin(np.pi * x)
    basis = hat_basis()
    hierarchical = hierarchize(nodal_values, wavelet=basis)
    epsilon = 0.02
    compressed = compress(hierarchical, epsilon=epsilon)
    assert len(compressed.subspaces) < len(hierarchical.subspaces)
    # dropping the finest subspaces shrinks max_level; compare via interpolation at the
    # original nodes instead
    coords = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1).reshape(-1, 2)
    values = interpolate(coords, compressed, wavelet=basis)
    assert np.max(np.abs(values - nodal_values.ravel())) <= epsilon * len(
        hierarchical.subspaces
    )


@pytest.mark.parametrize("basis", [hat_basis(), cdf_2_2_basis(Neumann(), Neumann())])
def test_linear_bases_reproduce_nodal_values(basis):
    x = np.linspace(0, 1, 17)
    values = np.sin(2 * np.pi * x) + x
    hierarchical = hierarchize(values, wavelet=basis)
    at_nodes = [interpolate(np.array([c]), hierarchical, wavelet=basis) for c in x]
    assert np.allclose(at_nodes, values, atol=1e-12)
    # between nodes, the hat basis interpolates linearly
    between = RNG.uniform(0, 1, size=8)
    expected = np.interp(between, x, values)
    at_between = [
        interpolate(np.array([c]), hierarchical, wavelet=basis) for c in between
    ]
    assert np.allclose(at_between, expected, atol=1e-12)


def test_interpolate_matches_dehierarchize_2d_mixed():
    nodal_values = RNG.normal(size=(8, 17))
    bases = (haar_basis(), cdf_2_2_basis(Neumann(), Neumann()))
    hierarchical = hierarchize(nodal_values, wavelet=bases, min_level=(1, 0))
    # at cell-midpoint x nodal-y coordinates: exact values
    x_mid = (np.arange(8) + 0.5) / 8
    y_nodes = np.arange(17) / 16
    coords = np.array([(x, y) for x in x_mid for y in y_nodes])
    values = interpolate(coords, hierarchical, wavelet=bases)
    assert np.allclose(values, nodal_values.ravel(), atol=1e-11)


@pytest.mark.parametrize(
    "g_left,g_right", [(0.0, 2.0), (1.0, 4.0)], ids=["half-homogeneous", "general"]
)
def test_interpolate_interior_only_uses_boundary_values(g_left, g_right):
    x = np.linspace(0, 1, 17)
    basis = hat_basis(Dirichlet(g_left), Dirichlet(g_right), include_boundary=False)
    values = g_left + (g_right - g_left) * x
    hierarchical = hierarchize(values[1:-1], wavelet=basis, min_level=1)
    coords = RNG.uniform(0, 1, size=8)
    interpolated = [
        interpolate(np.array([c]), hierarchical, wavelet=basis) for c in coords
    ]
    assert np.allclose(interpolated, g_left + (g_right - g_left) * coords, atol=1e-12)


def test_interpolate_after_compression_sparse_subspaces():
    nodal_values = RNG.normal(size=(17, 17))
    basis = hat_basis()
    hierarchical = hierarchize(nodal_values, wavelet=basis)
    compressed = compress(hierarchical, epsilon=1e-12)
    coords = RNG.uniform(0, 1, size=(8, 2))
    assert np.allclose(
        interpolate(coords, compressed, wavelet=basis),
        interpolate(coords, hierarchical, wavelet=basis),
        atol=1e-11,
    )


def test_cubic_interpolation_not_supported():
    values = RNG.normal(size=17)
    basis = cubic_basis()
    hierarchical = hierarchize(values, wavelet=basis)
    with pytest.raises(NotImplementedError, match="evaluation"):
        interpolate(np.array([0.3]), hierarchical, wavelet=basis)


def test_basis_validation():
    with pytest.raises(ValueError, match="cell-centered"):
        Basis1D(VertexCentered(True), haar_scheme)
    with pytest.raises(ValueError, match="Dirichlet"):
        Basis1D(VertexCentered(False), hierarchical_hat, Neumann(), Neumann())
    with pytest.raises(ValueError, match="[Pp]eriodic"):
        Basis1D(VertexCentered(True), hierarchical_hat, Periodic(), Neumann())
    with pytest.raises(ValueError, match="[Pp]eriodic"):
        Basis1D(VertexCentered(periodic=True), hierarchical_hat, Neumann(), Neumann())
    with pytest.raises(ValueError, match="boundary node"):
        VertexCentered(include_boundary=False, periodic=True)
    with pytest.raises(ValueError, match="rules"):
        hat_basis(Neumann(), Neumann(), periodic=True)
    with pytest.raises(ValueError, match="extent"):
        hierarchize(RNG.normal(size=16), wavelet=hat_basis())


@pytest.mark.parametrize("make_basis", [hat_basis, cdf_2_2_basis, cubic_basis])
# multiples of the coarsest grid spacing, so every subspace's dof set maps
# onto itself
@pytest.mark.parametrize("shift", [8, 16], ids=["quarter-period", "half-period"])
def test_periodic_vertex_translation_equivariance(make_basis, shift):
    n = 32
    x = np.arange(n) / n
    f = np.exp(np.sin(2 * np.pi * x))
    min_level = 2
    basis = make_basis(periodic=True)

    reference = hierarchize(f, wavelet=basis, min_level=min_level)
    shifted = hierarchize(np.roll(f, shift), wavelet=basis, min_level=min_level)
    for level, subspace in reference.subspaces.items():
        coefficients = subspace.data.to_dense()
        expected = np.roll(coefficients, shift * len(coefficients) // n)
        assert np.allclose(
            shifted.subspaces[level].data.to_dense(), expected, atol=1e-12
        ), level


@pytest.mark.parametrize("min_level", [2, 3, 4])
def test_periodic_cdf_2_2_preserves_periodic_mean_exactly(min_level):
    # with wrap ghosts, the mean-preserving update holds exactly at every
    # level; the free boundary only preserves the mean approximately
    n = 64
    x = np.arange(n) / n
    f = np.exp(np.sin(2 * np.pi * x))

    wrapped = (
        hierarchize(f, wavelet=cdf_2_2_basis(periodic=True), min_level=min_level)
        .subspaces[(min_level,)]
        .data.to_dense()
    )
    free = (
        hierarchize(np.append(f, f[0]), wavelet=cdf_2_2_basis(), min_level=min_level)
        .subspaces[(min_level,)]
        .data.to_dense()
    )
    assert np.isclose(np.mean(wrapped), np.mean(f), atol=1e-13, rtol=0.0)
    assert abs(np.mean(free[:-1]) - np.mean(f)) > 1e-5


def test_periodic_vertex_2d_roundtrip_and_interpolation():
    # nonperiodic hat in x, periodic cdf_2_2 in y (seam node stored once)
    values = RNG.normal(size=(17, 32))
    bases = (hat_basis(), cdf_2_2_basis(periodic=True))
    hierarchical = hierarchize(values, wavelet=bases)
    assert np.allclose(dehierarchize(hierarchical, wavelet=bases), values, atol=1e-11)
    x_nodes = np.arange(17) / 16
    y_nodes = np.arange(32) / 32
    coords = np.array([(x, y) for x in x_nodes for y in y_nodes])
    interpolated = interpolate(coords, hierarchical, wavelet=bases)
    assert np.allclose(interpolated, values.ravel(), atol=1e-11)
    # evaluation past the last stored node wraps to the seam node
    wrap_coords = np.array([(x, 1.0) for x in x_nodes])
    assert np.allclose(
        interpolate(wrap_coords, hierarchical, wavelet=bases),
        values[:, 0],
        atol=1e-11,
    )
