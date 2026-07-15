# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""spght: Sparse Grid Hierarchical Tensors (pronounced "spaghetti").

Public API: `hierarchize` a function on a full grid into hierarchical
subspaces, `compress` it by dropping small coefficients, evaluate it with
`interpolate`, and `write`/`read` the .spght binary format (also available
as methods on `SparseGridHierarchicalTensors`).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from spght.basis import (
    BoundaryRule,
    CellCentered,
    Centering,
    Dirichlet,
    Extrapolate,
    Neumann,
    Periodic,
    VertexCentered,
)
from spght.compress import compress
from spght.data_structures import (
    MetadataValue,
    OpaqueValue,
    SparseGridHierarchicalTensors,
    Subspace,
    subspace_order_key,
)
from spght.hierarchize import dehierarchize, hierarchize
from spght.interpolate import interpolate
from spght.lifting import Basis1D, LiftingScheme, LiftingStep
from spght.wavelets import (
    cdf_2_2_basis,
    cubic_basis,
    haar_basis,
    hat_basis,
)
from spght.linearize import Order, midpoint_coordinates_from_level
from spght.serialize import read, write
from spght.tensor import DenseTensor, SparseTensor, Tensor, TensorKind

try:
    __version__ = _version("spght")
except PackageNotFoundError:  # not installed, e.g. running from a checkout
    __version__ = "unknown"

__all__ = [
    "MetadataValue",
    "OpaqueValue",
    "Basis1D",
    "BoundaryRule",
    "CellCentered",
    "Centering",
    "DenseTensor",
    "Dirichlet",
    "Extrapolate",
    "LiftingScheme",
    "LiftingStep",
    "Neumann",
    "Order",
    "Periodic",
    "SparseGridHierarchicalTensors",
    "SparseTensor",
    "Subspace",
    "Tensor",
    "TensorKind",
    "VertexCentered",
    "__version__",
    "cdf_2_2_basis",
    "compress",
    "cubic_basis",
    "dehierarchize",
    "haar_basis",
    "hat_basis",
    "hierarchize",
    "interpolate",
    "midpoint_coordinates_from_level",
    "read",
    "subspace_order_key",
    "write",
]
