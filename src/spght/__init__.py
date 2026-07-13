# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#

"""spght: Sparse Grid Hierarchical Tensors (pronounced "spaghetti").

Public API: `hierarchize` a function on a full grid into hierarchical
subspaces, `compress` it by dropping small coefficients, evaluate it with
`interpolate`, and `write`/`read` the .spght binary format (also available
as methods on `SparseGridHierarchicalTensors`).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from spght.compress import compress
from spght.data_structures import (
    SparseGridHierarchicalTensors,
    Subspace,
    subspace_order_key,
)
from spght.hierarchize import hierarchize
from spght.interpolate import interpolate
from spght.linearize import Order, midpoint_coordinates_from_level
from spght.serialize import read, write
from spght.tensor import DenseTensor, SparseTensor, Tensor, TensorKind

try:
    __version__ = _version("spght")
except PackageNotFoundError:  # not installed, e.g. running from a checkout
    __version__ = "unknown"

__all__ = [
    "DenseTensor",
    "Order",
    "SparseGridHierarchicalTensors",
    "SparseTensor",
    "Subspace",
    "Tensor",
    "TensorKind",
    "__version__",
    "compress",
    "hierarchize",
    "interpolate",
    "midpoint_coordinates_from_level",
    "read",
    "subspace_order_key",
    "write",
]
