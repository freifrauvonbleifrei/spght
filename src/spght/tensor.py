# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Prototype API for an adaptive hierarchical sparse grid file format.
The binary encoding details may still evolve.
"""

import abc
from enum import IntEnum
import numpy as np
import numpy.typing as npt
from typing import ClassVar, Iterator

from spght.linearize import (
    IndexLike,
    Order,
    indices_to_multidim_indices,
    multidim_indices_to_indices,
)


class TensorKind(IntEnum):
    """Storage kind of a tensor's linear buffer. Enumerable so that further
    tensor implementations can be added later; the numeric value is meant to
    become the on-disk index-kind identifier in the subspace header."""

    EMPTY = 0  # no stored values, all entries implicitly zero (the default)
    FULL = 1  # dense: complete value buffer, no index list
    POINTWISE = 2  # sparse: individual sorted linear indices + matching values
    INTERVALS = 3  # sparse: maximal runs of consecutive linear indices + values


class Tensor(abc.ABC):
    """Logical n-d array stored as a linear buffer in `order` linearization.

    Indexing contract (via __getitem__/__setitem__):
      - int (or 0-d array): single LINEAR index -> scalar
      - tuple of ints: single MULTIDIM index -> scalar
      - 1-D array/sequence: batch of LINEAR indices -> 1-D array
      - 2-D array (n x ndim): batch of MULTIDIM indices -> 1-D array
    Negative (python-style) indices are not supported.
    """

    #: storage kind; every concrete Tensor implementation must set this
    kind: ClassVar[TensorKind]

    shape: tuple[int, ...]
    dtype: np.dtype
    order: Order

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def is_sparse(self) -> bool:
        """Derived from kind: anything that stores an index list is sparse."""
        return self.kind != TensorKind.FULL

    @property
    @abc.abstractmethod
    def nnz(self) -> int: ...

    @property
    @abc.abstractmethod
    def linear_values(self) -> npt.NDArray:
        """The canonical 1-D value buffer, in `order` linearization.
        This is what a file's subspace data block would contain."""
        ...

    @property
    @abc.abstractmethod
    def linear_indices(self) -> npt.NDArray[np.int64]:
        """Sorted linear indices of the stored entries, matching
        linear_values; for a dense tensor that is simply all positions."""
        ...

    @property
    @abc.abstractmethod
    def nbytes(self) -> int:
        """In-memory storage size: value buffer plus (if sparse) index buffer."""
        ...

    @abc.abstractmethod
    def _get_linear(self, linear: npt.NDArray[np.int64]) -> npt.NDArray: ...

    @abc.abstractmethod
    def _set_linear(
        self, linear: npt.NDArray[np.int64], values: npt.NDArray
    ) -> None: ...

    @abc.abstractmethod
    def to_dense(self) -> npt.NDArray:
        """Materialize the full n-d numpy array (a fresh copy)."""
        ...

    @abc.abstractmethod
    def with_order(self, order: Order) -> "Tensor":
        """The same logical array linearized in `order`. Returns self
        (sharing the buffer) when the order already matches, a rearranged
        copy otherwise."""
        ...

    @abc.abstractmethod
    def nonzero_items(self) -> Iterator[tuple[tuple[int, ...], float]]:
        """Yield ((multidim index), value) for stored nonzeros, in ascending
        linear-index order."""
        ...

    def _check_multidim_bounds(self, coords: npt.NDArray[np.int64]) -> None:
        extents = np.array(self.shape, dtype=np.int64)
        if np.any(coords < 0) or np.any(coords >= extents):
            raise IndexError(
                f"Multidim index out of bounds for shape {self.shape}: {coords}"
            )

    def _check_linear_bounds(self, linear: npt.NDArray[np.int64]) -> None:
        if np.any(linear < 0) or np.any(linear >= self.size):
            raise IndexError(f"Linear index out of bounds for size {self.size}")

    def _normalize_index(self, idx: IndexLike) -> tuple[npt.NDArray[np.int64], bool]:
        """Return (linear_indices, was_scalar)."""
        if isinstance(idx, (int, np.integer)):
            linear = np.array([idx], dtype=np.int64)
            self._check_linear_bounds(linear)
            return linear, True

        if isinstance(idx, tuple):
            if any(isinstance(c, slice) for c in idx):
                raise NotImplementedError(
                    "Slicing isn't supported; use to_dense() first."
                )
            coords = np.asarray(idx, dtype=np.int64).reshape(1, -1)
            if coords.shape[1] != self.ndim:
                raise ValueError(
                    f"Expected a multidim index of length {self.ndim}, got {coords.shape[1]}"
                )
            self._check_multidim_bounds(coords)
            return multidim_indices_to_indices(coords, self.shape, self.order), True

        arr = np.asarray(idx)
        if arr.ndim == 0:
            linear = arr.reshape(1).astype(np.int64)
            self._check_linear_bounds(linear)
            return linear, True
        if arr.ndim == 1:  # batch of LINEAR indices
            linear = arr.astype(np.int64)
            self._check_linear_bounds(linear)
            return linear, False
        if arr.ndim == 2:  # batch of MULTIDIM indices
            if arr.shape[1] != self.ndim:
                raise ValueError(
                    f"Expected multidim indices of width {self.ndim}, got {arr.shape[1]}"
                )
            coords = arr.astype(np.int64)
            self._check_multidim_bounds(coords)
            return (
                multidim_indices_to_indices(coords, self.shape, self.order),
                False,
            )
        raise ValueError(f"Unsupported index with ndim={arr.ndim}")

    def __getitem__(self, idx: IndexLike):
        linear, scalar = self._normalize_index(idx)
        values = self._get_linear(linear)
        return values[0] if scalar else values

    def __setitem__(self, idx: IndexLike, value) -> None:
        linear, scalar = self._normalize_index(idx)
        values = np.asarray(value, dtype=self.dtype)
        if values.ndim == 0:
            values = np.full(
                linear.shape[0], values, dtype=self.dtype
            )  # broadcast scalar over batch
        elif values.shape[0] != linear.shape[0]:
            raise ValueError(
                f"Got {values.shape[0]} values for {linear.shape[0]} indices"
            )
        self._set_linear(linear, values)

    def __eq__(self, other: object) -> bool:
        """Semantic equality: same logical array, order, and dtype --
        regardless of the storage kind (a dense and a sparse tensor holding
        the same values compare equal). Tensors are mutable and therefore
        unhashable."""
        if not isinstance(other, Tensor):
            return NotImplemented
        return (
            tuple(self.shape) == tuple(other.shape)
            and self.order == other.order
            and self.dtype == other.dtype
            and bool(np.array_equal(self.to_dense(), other.to_dense()))
        )

    __hash__ = None  # type: ignore[assignment]

    def __array__(self) -> npt.NDArray:
        return self.to_dense()

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} shape={self.shape} dtype={self.dtype} "
            f"order={self.order} {self.kind.name} nnz={self.nnz}>"
        )


class DenseTensor(Tensor):
    kind = TensorKind.FULL

    def __init__(self, flat: npt.NDArray, shape: tuple[int, ...], order: Order = "C"):
        total = int(np.prod(shape)) if shape else 1
        if flat.ndim != 1 or flat.shape[0] != total:
            raise ValueError(
                f"Expected a 1-D buffer of length {total} for shape {shape}, "
                f"got shape {flat.shape}"
            )
        self._flat = flat
        self.shape = shape
        self.dtype = flat.dtype
        self.order = order

    @classmethod
    def from_dense(cls, array: npt.NDArray, order: Order = "C") -> "DenseTensor":
        """Linearize an n-d array into the given order."""
        array = np.asarray(array)
        shape = array.shape
        if order == "C":
            flat = array.ravel(order="C").copy()
        elif order == "F":
            flat = array.ravel(order="F").copy()
        else:
            # gather: buffer position i holds the value at the multidim
            # coordinate that linearizes to i
            total = int(np.prod(shape)) if shape else 1
            coords = indices_to_multidim_indices(np.arange(total), shape, order)
            flat = array[tuple(coords.T)]
        return cls(flat, shape, order=order)

    @property
    def nnz(self) -> int:
        return int(np.count_nonzero(self._flat))

    @property
    def linear_values(self) -> npt.NDArray:
        return self._flat

    @property
    def linear_indices(self) -> npt.NDArray[np.int64]:
        """All positions: a dense tensor stores every entry."""
        return np.arange(self.size, dtype=np.int64)

    @property
    def nbytes(self) -> int:
        return int(self._flat.nbytes)

    def _get_linear(self, linear: npt.NDArray[np.int64]) -> npt.NDArray:
        return self._flat[linear]

    def _set_linear(self, linear: npt.NDArray[np.int64], values: npt.NDArray) -> None:
        self._flat[linear] = values

    def to_dense(self) -> npt.NDArray:
        if self.order in ("C", "F"):
            return self._flat.reshape(self.shape, order=self.order).copy()
        dense = np.empty(self.shape, dtype=self.dtype)
        coords = indices_to_multidim_indices(
            np.arange(self._flat.shape[0]), self.shape, self.order
        )
        dense[tuple(coords.T)] = self._flat
        return dense

    def with_order(self, order: Order) -> "DenseTensor":
        if order == self.order:
            return self
        # gather: for each target position, the source position holding
        # the value of the same multidim coordinate
        coords = indices_to_multidim_indices(np.arange(self.size), self.shape, order)
        source = multidim_indices_to_indices(coords, self.shape, self.order)
        return DenseTensor(self._flat[source], self.shape, order=order)

    def nonzero_items(self):
        nz = np.flatnonzero(self._flat)
        coords = indices_to_multidim_indices(nz, self.shape, self.order)
        for c, v in zip(coords, self._flat[nz]):
            yield tuple(int(i) for i in c), v


class SparseTensor(Tensor):
    """Sorted linear indices + matching values."""

    kind = TensorKind.POINTWISE

    def __init__(
        self,
        indices: npt.NDArray,
        values: npt.NDArray,
        shape: tuple[int, ...],
        order: Order = "C",
        pending_limit: int = 1024,
    ):
        keys = multidim_indices_to_indices(indices, shape, order)
        self._init_from_linear(keys, np.asarray(values), shape, order, pending_limit)

    @classmethod
    def from_linear(
        cls,
        keys: npt.NDArray,
        values: npt.NDArray,
        shape: tuple[int, ...],
        order: Order = "C",
        pending_limit: int = 1024,
    ) -> "SparseTensor":
        """Construct directly from linear indices (need not be sorted)."""
        self = cls.__new__(cls)
        self._init_from_linear(
            np.asarray(keys), np.asarray(values), shape, order, pending_limit
        )
        return self

    @classmethod
    def from_dense(
        cls,
        array: npt.NDArray,
        order: Order = "C",
        pending_limit: int = 1024,
    ) -> "SparseTensor":
        """Linearize an n-d array and keep only its nonzero entries."""
        flat = DenseTensor.from_dense(array, order=order).linear_values
        keys = np.flatnonzero(flat)
        return cls.from_linear(
            keys, flat[keys], np.asarray(array).shape, order, pending_limit
        )

    def _init_from_linear(self, keys, values, shape, order, pending_limit) -> None:
        if keys.shape != values.shape or keys.ndim != 1:
            raise ValueError(
                f"Expected matching 1-D keys and values, got {keys.shape} and {values.shape}"
            )
        self.shape = shape
        self.dtype = values.dtype
        self.order = order
        self._pending_limit = pending_limit
        self._pending: dict[int, float] = {}

        keys = keys.astype(np.int64)
        total = int(np.prod(shape)) if shape else 1
        if keys.size and (keys.min() < 0 or keys.max() >= total):
            # invalid keys would otherwise surface only much later (or, on
            # serialization, silently wrap in the narrowing index cast)
            raise ValueError(
                f"Linear keys must be within [0, {total}) for shape {shape}"
            )
        sort_order = np.argsort(keys)
        self._keys = keys[sort_order]
        self._values = values[sort_order]
        if np.any(self._keys[1:] == self._keys[:-1]):
            raise ValueError("Linear keys must be unique")

    @property
    def nnz(self) -> int:
        return len(self._values) + len(self._pending)

    @property
    def linear_values(self) -> npt.NDArray:
        self._merge_pending()
        return self._values

    @property
    def linear_indices(self) -> npt.NDArray[np.int64]:
        """Sorted linear indices matching linear_values."""
        self._merge_pending()
        return self._keys

    @property
    def nbytes(self) -> int:
        self._merge_pending()
        return int(self._values.nbytes + self._keys.nbytes)

    def _merge_pending(self) -> None:
        if not self._pending:
            return
        new_keys = np.fromiter(
            self._pending.keys(), dtype=self._keys.dtype, count=len(self._pending)
        )
        new_values = np.fromiter(
            self._pending.values(), dtype=self._values.dtype, count=len(self._pending)
        )
        merged_keys = np.concatenate([self._keys, new_keys])
        merged_values = np.concatenate([self._values, new_values])
        order = np.argsort(merged_keys)
        self._keys, self._values = merged_keys[order], merged_values[order]
        self._pending.clear()

    def _find(self, linear: npt.NDArray[np.int64]):
        """Return (positions clipped into range, found mask) in the sorted keys."""
        n_keys = len(self._keys)
        if n_keys == 0:
            zeros = np.zeros(linear.shape[0], dtype=np.int64)
            return zeros, np.zeros(linear.shape[0], dtype=bool)
        pos = np.searchsorted(self._keys, linear)
        pos_clipped = np.clip(pos, 0, n_keys - 1)
        found = (pos < n_keys) & (self._keys[pos_clipped] == linear)
        return pos_clipped, found

    def _get_linear(self, linear: npt.NDArray[np.int64]) -> npt.NDArray:
        pos_clipped, found = self._find(linear)
        result = np.zeros(linear.shape[0], dtype=self.dtype)
        result[found] = self._values[pos_clipped[found]]

        if self._pending:  # bounded by pending_limit, so this loop is cheap
            for i in np.flatnonzero(~found):
                v = self._pending.get(int(linear[i]))
                if v is not None:
                    result[i] = v
        return result

    def _set_linear(self, linear: npt.NDArray[np.int64], values: npt.NDArray) -> None:
        if linear.shape[0] == 1:
            self._set_single(int(linear[0]), values[0])
        else:
            self._set_batch(linear, values)

    def _set_single(self, key: int, value) -> None:
        n_keys = len(self._keys)
        pos = int(np.searchsorted(self._keys, key))
        found = pos < n_keys and self._keys[pos] == key

        if found:
            if value == 0:
                self._keys = np.delete(self._keys, pos)
                self._values = np.delete(self._values, pos)
            else:
                self._values[pos] = value
        elif value != 0:
            self._pending[key] = value
            if len(self._pending) >= self._pending_limit:
                self._merge_pending()
        else:
            self._pending.pop(key, None)

    def _set_batch(self, linear: npt.NDArray[np.int64], values: npt.NDArray) -> None:
        self._merge_pending()

        # same index written twice in one batch -> last write wins
        order = np.argsort(linear, kind="stable")
        linear, values = linear[order], values[order]
        is_last = np.empty(len(linear), dtype=bool)
        is_last[:-1] = linear[1:] != linear[:-1]
        is_last[-1] = True
        linear, values = linear[is_last], values[is_last]

        pos_clipped, found = self._find(linear)

        update_mask = found & (values != 0)
        self._values[pos_clipped[update_mask]] = values[update_mask]

        delete_mask = found & (values == 0)
        if np.any(delete_mask):
            keep = np.ones(len(self._keys), dtype=bool)
            keep[pos_clipped[delete_mask]] = False
            self._keys, self._values = self._keys[keep], self._values[keep]

        insert_mask = (~found) & (values != 0)
        if np.any(insert_mask):
            merged_keys = np.concatenate([self._keys, linear[insert_mask]])
            merged_values = np.concatenate([self._values, values[insert_mask]])
            order2 = np.argsort(merged_keys)
            self._keys, self._values = merged_keys[order2], merged_values[order2]

    def to_dense(self) -> npt.NDArray:
        self._merge_pending()
        dense = np.zeros(self.shape, dtype=self.dtype)
        coords = indices_to_multidim_indices(self._keys, self.shape, self.order)
        dense[tuple(coords.T)] = self._values
        return dense

    def nonzero_items(self):
        self._merge_pending()
        coords = indices_to_multidim_indices(self._keys, self.shape, self.order)
        for c, v in zip(coords, self._values):
            yield tuple(int(i) for i in c), v

    def with_order(self, order: Order) -> "SparseTensor":
        if order == self.order:
            return self
        # remap the stored keys only: O(nnz), never materializes the array
        coords = indices_to_multidim_indices(
            self.linear_indices, self.shape, self.order
        )
        keys = multidim_indices_to_indices(coords, self.shape, order)
        return SparseTensor.from_linear(
            keys, self.linear_values.copy(), self.shape, order=order
        )


def runs_from_sorted_keys(
    keys: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Split sorted unique linear indices into maximal runs of consecutive
    positions; returns (firsts, lasts), both ends inclusive."""
    if keys.size == 0:
        return keys, keys
    breaks = np.flatnonzero(np.diff(keys) > 1)
    firsts = keys[np.concatenate(([0], breaks + 1))]
    lasts = keys[np.concatenate((breaks, [keys.size - 1]))]
    return firsts, lasts


class IntervalTensor(Tensor):
    """Maximal runs of consecutive linear indices + the values they cover.

    The index structure is a sorted list of disjoint runs [first, last]
    (both inclusive) of linear positions in `order` linearization,
    separated by at least one uncovered position (maximal, so the
    representation is canonical). Every covered position stores a value,
    every other position is implicitly zero.

    The runs are structural: values inside runs can be updated (including
    to zero, which keeps the run), but writing outside the covered set is
    not supported -- convert to a SparseTensor for that."""

    kind = TensorKind.INTERVALS

    def __init__(
        self,
        firsts: npt.NDArray,
        lasts: npt.NDArray,
        values: npt.NDArray,
        shape: tuple[int, ...],
        order: Order = "C",
    ):
        firsts = np.asarray(firsts, dtype=np.int64)
        lasts = np.asarray(lasts, dtype=np.int64)
        values = np.asarray(values)
        if firsts.ndim != 1 or firsts.shape != lasts.shape:
            raise ValueError(
                f"Expected matching 1-D run bounds, got {firsts.shape} "
                f"and {lasts.shape}"
            )
        if values.ndim != 1:
            raise ValueError(f"Expected a 1-D value buffer, got shape {values.shape}")
        total = int(np.prod(shape)) if shape else 1
        if np.any(firsts < 0) or np.any(lasts >= total):
            raise ValueError(f"Runs must lie within [0, {total}) for shape {shape}")
        if np.any(lasts < firsts):
            raise ValueError("Each run needs last >= first")
        if np.any(firsts[1:] <= lasts[:-1] + 1):
            raise ValueError(
                "Runs must be sorted, disjoint, and maximal (at least one "
                "uncovered position between consecutive runs)"
            )
        lengths = lasts - firsts + 1
        if int(lengths.sum()) != values.shape[0]:
            raise ValueError(
                f"Runs cover {int(lengths.sum())} positions, "
                f"got {values.shape[0]} values"
            )
        self.shape = shape
        self.dtype = values.dtype
        self.order = order
        self._firsts = firsts
        self._lasts = lasts
        self._values = values
        # position of each run's first value in the value buffer
        self._offsets = np.concatenate(([0], np.cumsum(lengths)))

    @classmethod
    def from_linear(
        cls,
        keys: npt.NDArray,
        values: npt.NDArray,
        shape: tuple[int, ...],
        order: Order = "C",
    ) -> "IntervalTensor":
        """Construct from unique linear indices (need not be sorted);
        consecutive indices coalesce into runs."""
        keys = np.asarray(keys, dtype=np.int64)
        values = np.asarray(values)
        if keys.shape != values.shape or keys.ndim != 1:
            raise ValueError(
                f"Expected matching 1-D keys and values, got {keys.shape} "
                f"and {values.shape}"
            )
        sort_order = np.argsort(keys)
        keys, values = keys[sort_order], values[sort_order]
        if np.any(keys[1:] == keys[:-1]):
            raise ValueError("Linear keys must be unique")
        firsts, lasts = runs_from_sorted_keys(keys)
        return cls(firsts, lasts, values, shape, order=order)

    @classmethod
    def from_dense(cls, array: npt.NDArray, order: Order = "C") -> "IntervalTensor":
        """Linearize an n-d array and keep runs of its nonzero entries."""
        flat = DenseTensor.from_dense(array, order=order).linear_values
        keys = np.flatnonzero(flat)
        return cls.from_linear(keys, flat[keys], np.asarray(array).shape, order)

    @property
    def num_runs(self) -> int:
        return int(self._firsts.size)

    @property
    def nnz(self) -> int:
        """Number of stored (covered) entries."""
        return int(self._values.size)

    @property
    def linear_values(self) -> npt.NDArray:
        return self._values

    @property
    def linear_indices(self) -> npt.NDArray[np.int64]:
        """The covered positions, expanded from the runs (sorted)."""
        if self._firsts.size == 0:
            return np.empty(0, dtype=np.int64)
        lengths = self._lasts - self._firsts + 1
        # value buffer position j lies in run r at index
        # firsts[r] + (j - offsets[r])
        return np.arange(self._offsets[-1]) + np.repeat(
            self._firsts - self._offsets[:-1], lengths
        )

    @property
    def nbytes(self) -> int:
        return int(self._values.nbytes + self._firsts.nbytes + self._lasts.nbytes)

    def _run_positions(self, linear: npt.NDArray[np.int64]):
        """Return (value buffer positions, covered mask) for linear indices."""
        if self._firsts.size == 0:
            zeros = np.zeros(linear.shape[0], dtype=np.int64)
            return zeros, np.zeros(linear.shape[0], dtype=bool)
        run = np.searchsorted(self._firsts, linear, side="right") - 1
        run_clipped = np.maximum(run, 0)
        covered = (run >= 0) & (linear <= self._lasts[run_clipped])
        positions = self._offsets[run_clipped] + (linear - self._firsts[run_clipped])
        return positions, covered

    def _get_linear(self, linear: npt.NDArray[np.int64]) -> npt.NDArray:
        positions, covered = self._run_positions(linear)
        result = np.zeros(linear.shape[0], dtype=self.dtype)
        result[covered] = self._values[positions[covered]]
        return result

    def _set_linear(self, linear: npt.NDArray[np.int64], values: npt.NDArray) -> None:
        positions, covered = self._run_positions(linear)
        if not np.all(covered):
            raise NotImplementedError(
                "IntervalTensor only supports writes inside its stored runs"
            )
        self._values[positions] = values

    def to_dense(self) -> npt.NDArray:
        dense = np.zeros(self.shape, dtype=self.dtype)
        if self._values.size:
            coords = indices_to_multidim_indices(
                self.linear_indices, self.shape, self.order
            )
            dense[tuple(coords.T)] = self._values
        return dense

    def nonzero_items(self):
        keys = self.linear_indices
        nonzero = np.flatnonzero(self._values)
        coords = indices_to_multidim_indices(keys[nonzero], self.shape, self.order)
        for c, v in zip(coords, self._values[nonzero]):
            yield tuple(int(i) for i in c), v

    def with_order(self, order: Order) -> "IntervalTensor":
        if order == self.order:
            return self
        coords = indices_to_multidim_indices(
            self.linear_indices, self.shape, self.order
        )
        keys = multidim_indices_to_indices(coords, self.shape, order)
        # from_linear re-derives the runs of the new linearization
        return IntervalTensor.from_linear(keys, self._values, self.shape, order=order)


def make_tensor_from_linear(
    keys, values, shape, order: Order = "C", density_threshold: float = 0.5
) -> Tensor:
    """Pick a representation by density, from linear indices + values."""
    keys = np.asarray(keys)
    values = np.asarray(values)
    total = int(np.prod(shape)) if shape else 1
    density = len(values) / total if total else 0.0
    if density < density_threshold:
        return SparseTensor.from_linear(keys, values, shape, order=order)
    flat = np.zeros(total, dtype=values.dtype)
    flat[keys] = values
    return DenseTensor(flat, shape, order=order)


def make_tensor(
    indices, values, shape, order: Order = "C", density_threshold: float = 0.5
) -> Tensor:
    """Pick a representation by density, from multidim indices + values."""
    keys = multidim_indices_to_indices(indices, shape, order)
    return make_tensor_from_linear(keys, values, shape, order, density_threshold)
