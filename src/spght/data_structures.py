"""Prototype API for an adaptive hierarchical sparse grid file format.
The binary encoding details may still evolve.
"""

import abc
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import BinaryIO, ClassVar, Iterator, Sequence, cast

from spght.linearize import (
    IndexLike,
    Order,
    indices_to_multidim_indices,
    multidim_indices_to_indices,
)

FormatMagic = b"sparse grid hierarchical tensors\0"
FormatVersion: tuple[int, int] = (0, 1)


class TensorKind(IntEnum):
    """Storage kind of a tensor's linear buffer. Enumerable so that further
    tensor implementations can be added later; the numeric value is meant to
    become the on-disk index-kind identifier in the subspace header."""

    EMPTY = 0  # no stored values, all entries implicitly zero (the default)
    FULL = 1  # dense: complete value buffer, no index list
    LINEAR = 2  # sparse: sorted single linear indices + matching values
    # future, e.g.: INTERVALS = 3  (runs of linear indices)


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
    def density(self) -> float:
        return self.nnz / self.size if self.size else 0.0

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

    def nonzero_items(self):
        nz = np.flatnonzero(self._flat)
        coords = indices_to_multidim_indices(nz, self.shape, self.order)
        for c, v in zip(coords, self._flat[nz]):
            yield tuple(int(i) for i in c), v


class SparseTensor(Tensor):
    """Sorted linear indices + matching values."""

    kind = TensorKind.LINEAR

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
        sort_order = np.argsort(keys)
        self._keys = keys[sort_order]
        self._values = values[sort_order]

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


@dataclass(frozen=True, slots=True)
class Subspace:
    """Describe one logical subspace in memory and on disk."""

    extents: tuple[int, ...]
    precision_bits: int
    data: Tensor | None = None
    padding_bits: int = 0
    checksum: int = 0
    compression: int = 0

    @property
    def order(self) -> Order:
        return self.data.order if self.data is not None else "C"

    @property
    def kind(self) -> TensorKind:
        """Storage kind of the subspace data (EMPTY when no data is attached)."""
        return self.data.kind if self.data is not None else TensorKind.EMPTY

    @property
    def is_sparse(self) -> bool:
        return self.data.is_sparse if self.data is not None else False

    @property
    def values(self) -> npt.NDArray | None:
        """Backward-compatible n-d view of the data (a fresh copy)."""
        return self.data.to_dense() if self.data is not None else None

    @property
    def num_bytes(self) -> int:
        return self.data.nbytes if self.data is not None else 0

    def __post_init__(self) -> None:
        if self.precision_bits <= 0:
            raise ValueError("precision_bits must be positive")
        if self.padding_bits < 0:
            raise ValueError("padding_bits must not be negative")
        if self.data is not None and tuple(self.data.shape) != tuple(self.extents):
            raise ValueError(
                f"data shape {self.data.shape} does not match extents {self.extents}"
            )


def subspace_order_key(level: Sequence[int]) -> tuple[int, tuple[int, ...]]:
    """Canonical subspace ordering: ascending level sum (coarse to fine),
    ties broken lexicographically (dimension 0 most significant)."""
    level_tuple = tuple(level)
    return (sum(level_tuple), level_tuple)


@dataclass(slots=True)
class SparseGridHierarchicalTensors:
    magic = FormatMagic
    dimensions: int
    max_level: tuple[int, ...]
    # levels: tuple[int, ...]
    subspaces: dict[tuple[int, ...], Subspace] = field(default_factory=dict)
    version = FormatVersion

    def __post_init__(self) -> None:
        for k, v in self.subspaces.items():
            if len(k) != self.dimensions:
                raise ValueError("subspace key dimensionality does not match container")
            if len(v.extents) != self.dimensions:
                raise ValueError(
                    "subspace extents dimensionality does not match container"
                )
        self._sort_subspaces()

    def _sort_subspaces(self) -> None:
        """Restore the canonical (level sum, lexicographic) subspace order."""
        self.subspaces = dict(
            sorted(self.subspaces.items(), key=lambda item: subspace_order_key(item[0]))
        )

    def add_subspace(self, level: tuple[int, ...], subspace: Subspace) -> None:
        if len(level) != self.dimensions:
            raise ValueError("subspace dimensionality does not match container")
        if len(self.max_level) != self.dimensions:
            raise ValueError("max_level dimensionality does not match container")
        self.subspaces[level] = subspace
        self._sort_subspaces()


def open_file(path: str | Path, mode: str = "rb") -> BinaryIO:
    """Small wrapper for file access used by the prototype API."""

    return cast(BinaryIO, Path(path).open(mode))
