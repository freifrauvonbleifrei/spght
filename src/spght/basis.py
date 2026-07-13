# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Grid centerings and boundary rules for lifting-based wavelet transforms.

- `Centering`: where the degrees of freedom sit (cell midpoints, or vertices
  with/without the boundary nodes)
- `BoundaryRule`: how stencil taps that fall outside the domain are answered
  (Dirichlet value, Neumann mirror, periodic wrap, extrapolation).

The third axis, the `LiftingScheme` (the wavelet itself), lives in
`spght.lifting`.
"""

import abc
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

WallType = Literal["node", "half", "offset"]


def _take(arr: npt.NDArray, index: int, axis: int) -> npt.NDArray:
    """One slab of `arr` at `index` along `axis`, keeping the axis."""
    if not 0 <= index < arr.shape[axis]:
        raise ValueError(
            "boundary stencil reaches deeper into the domain than stored "
            f"(index {index} along axis of extent {arr.shape[axis]})"
        )
    return np.take(arr, [index], axis=axis)


class BoundaryRule(abc.ABC):
    @abc.abstractmethod
    def ghost_slab(
        self, arr: npt.NDArray, position: int, wall: WallType, axis: int
    ) -> npt.NDArray:
        """Values at the virtual index `position` (< 0 beyond the left wall,
        >= extent beyond the right wall), as a slab along `axis`."""

    def homogenized(self) -> "BoundaryRule":
        """The rule with any inhomogeneous data removed. Applied to detail
        (wavelet) channels, where boundary data belongs to the scaling part
        only; the default rules are already homogeneous."""
        return self

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(vars(self).items()))))


class Periodic(BoundaryRule):
    """Wrap-around extension. Note that a 'node' wall stores the seam value
    at both ends (arr[0] == arr[-1]), so the period is extent - 1 there."""

    def ghost_slab(self, arr, position, wall, axis):
        period = arr.shape[axis] - 1 if wall == "node" else arr.shape[axis]
        return _take(arr, position % period, axis)


class Neumann(BoundaryRule):
    """Mirror extension: zero normal derivative at the wall."""

    def ghost_slab(self, arr, position, wall, axis):
        extent = arr.shape[axis]
        if wall == "node":  # whole-sample symmetry about the wall dof
            index = -position if position < 0 else 2 * (extent - 1) - position
        else:  # half-sample symmetry about the wall face
            index = -position - 1 if position < 0 else 2 * extent - 1 - position
        return _take(arr, index, axis)


@dataclass(frozen=True, eq=False)
class Dirichlet(BoundaryRule):
    """Known wall value; ghosts by antisymmetric reflection about it."""

    value: float = 0.0

    def ghost_slab(self, arr, position, wall, axis):
        extent = arr.shape[axis]
        if wall == "offset":
            # the wall itself sits one dof spacing outside the array
            if position in (-1, extent):
                shape = list(arr.shape)
                shape[axis] = 1
                return np.full(shape, self.value)
            mirror = -position - 2 if position < 0 else 2 * extent - position
        elif wall == "node":
            # reflect antisymmetrically about the stored wall dof
            wall_slab = _take(arr, 0 if position < 0 else extent - 1, axis)
            mirror = -position if position < 0 else 2 * (extent - 1) - position
            return 2 * wall_slab - _take(arr, mirror, axis)
        else:  # "half"
            mirror = -position - 1 if position < 0 else 2 * extent - 1 - position
        return 2 * self.value - _take(arr, mirror, axis)

    def homogenized(self) -> "Dirichlet":
        return Dirichlet(0.0)


class Extrapolate(BoundaryRule):
    """Linear extrapolation from the two edge dofs ("free" boundary)."""

    def ghost_slab(self, arr, position, wall, axis):
        extent = arr.shape[axis]
        if extent == 1:
            return _take(arr, 0, axis)
        if position < 0:
            distance = -position
            edge, inner = _take(arr, 0, axis), _take(arr, 1, axis)
        else:
            distance = position - (extent - 1)
            edge, inner = _take(arr, extent - 1, axis), _take(arr, extent - 2, axis)
        return edge + distance * (edge - inner)


class Centering(abc.ABC):
    """Where the dofs of a 1D level live, and how a level splits into the
    surviving (even/coarse) and vanishing (odd/detail) dofs."""

    name: str
    # do the even (surviving) dofs come first in the fine-level array?
    even_first: bool
    # index shifts translating a scheme's canonical stencil offsets
    # (declared for vertex-with-boundary geometry) to this centering
    even_shift: int
    odd_shift: int
    # wall types of the even/odd sub-grids, for boundary-rule queries
    even_wall: WallType
    odd_wall: WallType
    # coarsest level the cascade may stop at (interior-only grids have no
    # level-0 dofs)
    lowest_min_level: int

    @abc.abstractmethod
    def num_dofs(self, level: int) -> int: ...

    @abc.abstractmethod
    def coordinates(self, level: int) -> npt.NDArray[np.float64]:
        """The 1D dof coordinates in [0, 1] at `level`."""

    def num_details(self, level: int) -> int:
        return self.num_dofs(level) - self.num_dofs(level - 1)

    def level_from_num_dofs(self, extent: int) -> int:
        for level in range(64):
            if self.num_dofs(level) == extent:
                return level
        raise ValueError(f"extent {extent} does not match any level of {self.name}")

    def level_from_num_details(self, extent: int) -> int:
        for level in range(1, 64):
            if self.num_details(level) == extent:
                return level
        raise ValueError(
            f"detail extent {extent} does not match any level of {self.name}"
        )

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and vars(self) == vars(other)

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(vars(self).items()))))


class CellCentered(Centering):
    """2**level cells with dofs at the midpoints."""

    name = "cell"
    even_first = True
    even_shift = 0
    odd_shift = 0
    even_wall: WallType = "half"
    odd_wall: WallType = "half"
    lowest_min_level = 0

    def num_dofs(self, level: int) -> int:
        return 2**level

    def coordinates(self, level: int) -> npt.NDArray[np.float64]:
        extent = self.num_dofs(level)
        return (np.arange(extent) + 0.5) / extent


class VertexCentered(Centering):
    """Dofs on the grid vertices i / 2**level"""

    odd_wall: WallType = "half"

    def __init__(self, include_boundary: bool = True):
        self.include_boundary = include_boundary
        self.name = "vertex" if include_boundary else "vertex-interior"
        self.even_first = include_boundary
        self.even_shift = 0 if include_boundary else -1
        self.odd_shift = 0 if include_boundary else 1
        self.even_wall: WallType = "node" if include_boundary else "offset"
        self.lowest_min_level = 0 if include_boundary else 1

    def num_dofs(self, level: int) -> int:
        return 2**level + 1 if self.include_boundary else 2**level - 1

    def coordinates(self, level: int) -> npt.NDArray[np.float64]:
        nodes = np.arange(2**level + 1) / 2**level
        return nodes if self.include_boundary else nodes[1:-1]
