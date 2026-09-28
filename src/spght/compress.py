# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
import heapq
from typing import Literal
import numpy as np
import numpy.typing as npt


from spght.basis import CellCentered
from spght.data_structures import SparseGridHierarchicalTensors
from spght.linearize import (
    indices_to_multidim_indices,
    multidim_indices_to_indices,
)
from spght.tensor import make_tensor_from_linear
from spght.wavelets import haar


def _haar_crown_masks(
    tensors: SparseGridHierarchicalTensors, epsilon: float, whole: bool
) -> dict[tuple[int, ...], npt.NDArray[np.bool_]]:
    """Propagate protection through the tensor-product Haar parent DAG.

    Pending nodes use C-order keys, independent of coefficient storage.
    Missing subspaces are traversed virtually, without allocating dense
    blocks. Each level is processed after all finer levels feeding it.
    """
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("crown compression requires a finite, nonnegative epsilon")
    if not all(
        isinstance(basis.centering, CellCentered) and basis.scheme.steps == haar.steps
        for basis in tensors.bases
    ):
        raise ValueError("crown compression currently supports only cell-centered Haar")

    minimum = tensors.min_level

    def shape_at(level: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(2 ** (l if l == m else l - 1) for l, m in zip(level, minimum))

    masks: dict[tuple[int, ...], npt.NDArray[np.bool_]] = {}
    pending: dict[tuple[int, ...], list[npt.NDArray[np.int64]]] = {}
    queue: list[tuple[int, tuple[int, ...]]] = []

    def enqueue(level: tuple[int, ...], keys: npt.NDArray[np.int64]) -> None:
        if not keys.size or level == minimum:
            return
        if level not in pending:
            pending[level] = []
            heapq.heappush(queue, (-sum(level), level))
        pending[level].append(keys)

    for level, subspace in tensors.subspaces.items():
        if subspace.extents != shape_at(level):
            raise ValueError(
                f"subspace {level} extents do not match Haar basis geometry"
            )
        data = subspace.data
        if data is None:
            masks[level] = np.zeros(0, dtype=bool)
            continue
        values = data.linear_values
        # Nonfinite values must not silently disappear or lose ancestors.
        keep = (np.abs(values) > epsilon) | ~np.isfinite(values)
        masks[level] = keep
        coords = indices_to_multidim_indices(
            data.linear_indices[keep], data.shape, data.order
        )
        enqueue(level, multidim_indices_to_indices(coords, data.shape, "C"))

    while queue:
        _, level = heapq.heappop(queue)
        shape = shape_at(level)
        keys = np.unique(np.concatenate(pending.pop(level)))
        coords = indices_to_multidim_indices(keys, shape, "C")
        stored_subspace = tensors.subspaces.get(level)
        if stored_subspace is not None and stored_subspace.data is not None:
            data = stored_subspace.data
            stored_keys = multidim_indices_to_indices(coords, shape, data.order)
            keep = masks[level]
            keep |= np.isin(data.linear_indices, stored_keys) & (
                data.linear_values != 0
            )
            if whole and keep.any():
                # All nonzeros in a retained block also protect their parents.
                extra = indices_to_multidim_indices(
                    data.linear_indices[data.linear_values != 0], shape, data.order
                )
                keys = np.union1d(keys, multidim_indices_to_indices(extra, shape, "C"))
                coords = indices_to_multidim_indices(keys, shape, "C")

        for d, (l, m) in enumerate(zip(level, minimum)):
            if l == m:
                continue
            parent = list(level)
            parent[d] -= 1
            parent_level = tuple(parent)
            parent_coords = coords.copy()
            # First details have the SAME support as their scaling parent.
            if l > m + 1:
                parent_coords[:, d] //= 2
            enqueue(
                parent_level,
                np.unique(
                    multidim_indices_to_indices(
                        parent_coords, shape_at(parent_level), "C"
                    )
                ),
            )
    return masks


def compress(
    hierarchical_tensors: SparseGridHierarchicalTensors,
    only_whole_subspaces: bool = False,
    epsilon: float = 0.0,
    density_threshold: float = 0.5,
    *,
    structure: Literal["independent", "crown"] = "independent",
) -> SparseGridHierarchicalTensors:
    """Drop hierarchical coefficients with |coefficient| <= epsilon.

    With ``structure="crown"`` (cell-centered Haar only), a coefficient
    can be zeroed only if every finer coefficient with overlapping support
    is zeroed too. Finer means componentwise greater levels, with at least
    one strict inequality. Missing entries are zero; protection propagates
    through missing levels. The all-scaling block is always preserved.
    Nonfinite coefficients are preserved in crown mode, which requires a
    finite, nonnegative epsilon. Whole-subspace mode also propagates the
    small nonzero coefficients of any block that must be retained.

    Subspaces that are kept unchanged are shared with the input container
    (not copied); mutating their coefficient data afterwards affects both
    containers."""
    if structure not in ("independent", "crown"):
        raise ValueError(f"unknown compression structure: {structure!r}")
    masks = (
        _haar_crown_masks(hierarchical_tensors, epsilon, only_whole_subspaces)
        if structure == "crown"
        else None
    )
    compressed_tensors = SparseGridHierarchicalTensors(
        dimensions=hierarchical_tensors.dimensions,
        max_level=hierarchical_tensors.max_level,
        min_level=hierarchical_tensors.min_level,
        bases=hierarchical_tensors.bases,
        metadata=hierarchical_tensors.metadata,
    )
    for level, subspace in hierarchical_tensors.subspaces.items():
        if subspace.data is None:  # EMPTY: implicitly all-zero
            coefficients = np.zeros(0)
        else:
            coefficients = subspace.data.linear_values
        keep = masks[level] if masks is not None else np.abs(coefficients) > epsilon
        if level == hierarchical_tensors.min_level:
            # always keep the all-scaling lmin subspace
            compressed_tensors.add_subspace(level, subspace)
        elif not keep.any():
            continue
        elif keep.all() or only_whole_subspaces:
            # kept unchanged: shared with the input container, not copied
            compressed_tensors.add_subspace(level, subspace)
        else:
            # partial compression: keep only the surviving coefficients
            assert subspace.data is not None
            keys = subspace.data.linear_indices[keep]
            compressed_data = make_tensor_from_linear(
                keys,
                coefficients[keep],
                subspace.data.shape,
                order=subspace.data.order,
                density_threshold=density_threshold,
            )
            compressed_tensors.add_subspace(
                level, replace(subspace, data=compressed_data)
            )

    # the maximum level is the elementwise maximum over the kept subspaces
    if compressed_tensors.subspaces:
        compressed_tensors.max_level = tuple(
            max(levels) for levels in zip(*compressed_tensors.subspaces.keys())
        )
    else:
        compressed_tensors.max_level = tuple(compressed_tensors.min_level)
    return compressed_tensors
