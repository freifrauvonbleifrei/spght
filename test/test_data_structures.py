import numpy as np
import pytest

from spght.data_structures import (
    SparseGridHierarchicalTensors,
    Subspace,
    TensorKind,
    subspace_order_key,
)
from spght.tensor import DenseTensor, SparseTensor, make_tensor


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_dense_from_dense_roundtrip(order):
    rng = np.random.default_rng(0)
    # power-of-two extents so that Z orders are valid too
    for shape in [(4,), (4, 2), (2, 4, 2)]:
        array = rng.random(shape)
        t = DenseTensor.from_dense(array, order=order)
        assert t.shape == shape
        assert not t.is_sparse
        assert t.linear_values.ndim == 1
        assert t.linear_values.shape[0] == array.size
        assert np.array_equal(t.to_dense(), array)


def test_dense_linear_layout_c_and_f():
    array = np.arange(8, dtype=np.float64).reshape(2, 4)
    assert np.array_equal(
        DenseTensor.from_dense(array, order="C").linear_values, array.ravel(order="C")
    )
    assert np.array_equal(
        DenseTensor.from_dense(array, order="F").linear_values, array.ravel(order="F")
    )


def test_dense_linear_layout_z_2x2():
    # for a 2x2 extent, one bit per dimension: ZC interleaves dim0 first
    # (== C order), ZF interleaves dim1 first (== F order)
    array = np.array([[0.0, 1.0], [2.0, 3.0]])
    assert np.array_equal(
        DenseTensor.from_dense(array, order="ZC").linear_values, array.ravel(order="C")
    )
    assert np.array_equal(
        DenseTensor.from_dense(array, order="ZF").linear_values, array.ravel(order="F")
    )


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_z_orders_support_arbitrary_extents(order):
    array = np.arange(3 * 2 * 5, dtype=np.float64).reshape(3, 2, 5)
    t = DenseTensor.from_dense(array, order=order)
    assert np.array_equal(np.sort(t.linear_values), np.sort(array.ravel()))
    assert np.array_equal(t.to_dense(), array)


def test_dense_get_set():
    array = np.arange(24, dtype=np.float64).reshape(4, 6)
    for order in ["C", "F"]:
        t = DenseTensor.from_dense(array, order=order)
        assert t[2, 3] == array[2, 3]  # tuple -> multidim
        assert np.array_equal(
            t[np.array([[0, 0], [3, 5]])], np.array([array[0, 0], array[3, 5]])
        )
        t[2, 3] = -1.0
        assert t[2, 3] == -1.0
        assert t.to_dense()[2, 3] == -1.0
        # 1-D index array = batch of linear indices into the buffer
        t[np.array([0, 1])] = np.array([-2.0, -3.0])
        assert np.array_equal(t.linear_values[:2], np.array([-2.0, -3.0]))


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_sparse_from_dense_roundtrip(order):
    rng = np.random.default_rng(1)
    shape = (4, 8, 2)
    array = np.zeros(shape)
    # sprinkle a few nonzeros
    for _ in range(6):
        array[tuple(rng.integers(0, s) for s in shape)] = rng.random() + 0.1
    t = SparseTensor.from_dense(array, order=order)
    assert t.is_sparse
    assert t.nnz == np.count_nonzero(array)
    assert np.array_equal(t.to_dense(), array)
    # keys are sorted and match values
    assert np.all(np.diff(t.linear_indices) > 0)
    assert t.linear_indices.shape == t.linear_values.shape


def test_sparse_get_set_semantics():
    indices = np.array([[0, 0, 0], [1, 2, 3], [3, 4, 5]])
    values = np.array([1.0, 2.0, 3.0])
    t = SparseTensor(indices, values, shape=(4, 5, 6), order="C")

    assert t[1, 2, 3] == 2.0  # stored entry
    assert t[1, 2, 4] == 0.0  # implicit zero
    assert t[0] == 1.0  # linear index of (0,0,0) in C order

    t[1, 2, 3] = 9.0  # overwrite stored
    assert t[1, 2, 3] == 9.0

    t[2, 2, 2] = 5.0  # new entry -> pending buffer
    assert t[2, 2, 2] == 5.0  # readable before merge
    assert t.nnz == 4

    t[1, 2, 3] = 0.0  # writing zero deletes
    assert t[1, 2, 3] == 0.0
    assert t.nnz == 3

    dense = t.to_dense()
    assert dense[2, 2, 2] == 5.0
    assert dense[1, 2, 3] == 0.0
    assert np.count_nonzero(dense) == 3


def test_sparse_batch_set_last_write_wins():
    t = SparseTensor.from_dense(np.zeros((4, 4)), order="C")
    # linear indices 5 written twice: last value wins
    t[np.array([5, 5, 7])] = np.array([1.0, 2.0, 3.0])
    assert t[5] == 2.0
    assert t[7] == 3.0
    assert t.nnz == 2
    # batch write zeros deletes
    t[np.array([5, 7])] = 0.0
    assert t.nnz == 0
    assert t[5] == 0.0


def test_sparse_pending_merge():
    t = SparseTensor.from_dense(np.zeros((8, 8)), order="C", pending_limit=4)
    for i in range(6):  # crosses the merge threshold
        t[np.int64(i)] = float(i + 1)
    assert t.nnz == 6
    assert np.array_equal(t.linear_indices, np.arange(6))
    assert np.array_equal(t.linear_values, np.arange(1.0, 7.0))


def test_sparse_empty_reads_zero():
    t = SparseTensor.from_dense(np.zeros((4, 4)), order="C")
    assert t.nnz == 0
    assert t[3, 3] == 0.0
    assert np.array_equal(t[np.array([0, 5, 15])], np.zeros(3))
    assert np.array_equal(t.to_dense(), np.zeros((4, 4)))


def test_out_of_bounds_indices_raise():
    dense = DenseTensor.from_dense(np.zeros((4, 5)), order="C")
    sparse = SparseTensor.from_dense(np.zeros((4, 5)), order="C")
    for t in [dense, sparse]:
        with pytest.raises(IndexError):
            t[4, 0]  # first extent is 4
        with pytest.raises(IndexError):
            t[0, 5]  # would silently alias a valid linear index otherwise
        with pytest.raises(IndexError):
            t[np.array([20])]  # linear out of range
        with pytest.raises(IndexError):
            t[-1, 0]  # negative indices unsupported


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_nonzero_items_sorted_and_correct(order):
    array = np.zeros((4, 4))
    array[0, 1] = 1.0
    array[2, 3] = 2.0
    array[3, 0] = 3.0
    for t in [
        DenseTensor.from_dense(array, order=order),
        SparseTensor.from_dense(array, order=order),
    ]:
        items = list(t.nonzero_items())
        assert {(c, v) for c, v in items} == {
            ((0, 1), 1.0),
            ((2, 3), 2.0),
            ((3, 0), 3.0),
        }


def test_make_tensor_density_pick():
    indices = np.array([[0, 0], [1, 1]])
    values = np.array([1.0, 2.0])
    # 2 of 64 entries -> sparse
    assert make_tensor(indices, values, shape=(8, 8)).is_sparse
    # 2 of 4 entries -> dense
    assert not make_tensor(indices, values, shape=(2, 2)).is_sparse


def test_tensor_nbytes():
    array = np.zeros((4, 4), dtype=np.float64)
    array[1, 1] = 1.0
    dense = DenseTensor.from_dense(array, order="C")
    assert dense.nbytes == 16 * 8
    sparse = SparseTensor.from_dense(array, order="C")
    assert sparse.nbytes == 8 + 8  # one float64 value + one int64 key


def test_tensor_kind():
    array = np.zeros((2, 2))
    dense = DenseTensor.from_dense(array)
    sparse = SparseTensor.from_dense(array)
    assert dense.kind == TensorKind.FULL and not dense.is_sparse
    assert sparse.kind == TensorKind.LINEAR and sparse.is_sparse
    # the kind is a class-level property of each implementation
    assert DenseTensor.kind == TensorKind.FULL
    assert SparseTensor.kind == TensorKind.LINEAR
    # numeric values are the (future) on-disk index-kind identifiers;
    # EMPTY is deliberately the zero value, so a zero-initialized header
    # reads as "no data"
    assert int(TensorKind.EMPTY) == 0
    assert int(TensorKind.FULL) == 1
    assert int(TensorKind.LINEAR) == 2
    # a subspace without data defaults to the EMPTY kind
    assert Subspace(extents=(2, 2), precision_bits=64).kind == TensorKind.EMPTY


def test_subspace_holds_tensor():
    array = np.arange(8, dtype=np.float64).reshape(2, 4)
    dense = DenseTensor.from_dense(array, order="C")
    subspace = Subspace(extents=(2, 4), precision_bits=64, data=dense)
    assert subspace.kind == TensorKind.FULL
    assert not subspace.is_sparse
    assert subspace.num_bytes == dense.nbytes
    assert np.array_equal(subspace.values, array)  # backward-compat shim

    sparse_sub = Subspace(
        extents=(2, 4),
        precision_bits=64,
        data=SparseTensor.from_dense(array, order="C"),
    )
    assert sparse_sub.kind == TensorKind.LINEAR
    assert sparse_sub.is_sparse

    with pytest.raises(ValueError):
        Subspace(extents=(4, 2), precision_bits=64, data=dense)  # shape mismatch


def test_hierarchize_produces_dense_tensors():
    from spght.hierarchize import hierarchize

    nodal_values = np.random.default_rng(2).random((4, 8))
    hierarchical_tensors = hierarchize(nodal_values)
    for _, subspace in hierarchical_tensors.subspaces.items():
        assert subspace.data is not None
        assert not subspace.data.is_sparse  # dense until compress() sparsifies
        assert subspace.data.linear_values.ndim == 1  # linear storage invariant
        assert tuple(subspace.data.shape) == tuple(subspace.extents)


def test_subspaces_canonically_ordered():
    def one_value_subspace():
        return Subspace(
            extents=(1, 1),
            precision_bits=64,
            data=DenseTensor.from_dense(np.ones((1, 1))),
        )

    shuffled_levels = [(2, 2), (0, 1), (2, 0), (0, 0), (1, 1), (0, 2), (1, 0)]
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(2, 2),
        subspaces={level: one_value_subspace() for level in shuffled_levels},
    )
    # ascending level sum, ties broken lexicographically
    assert list(tensors.subspaces.keys()) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (2, 2),
    ]

    # add_subspace re-establishes the canonical order
    tensors.add_subspace((1, 2), one_value_subspace())
    assert list(tensors.subspaces.keys()) == [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (1, 2),
        (2, 2),
    ]


def test_hierarchize_subspaces_canonically_ordered():
    from spght.hierarchize import hierarchize

    nodal_values = np.random.default_rng(4).random((4, 2, 4))
    hierarchical_tensors = hierarchize(nodal_values)
    keys = list(hierarchical_tensors.subspaces.keys())
    assert keys == sorted(keys, key=subspace_order_key)
    assert keys[0] == (0, 0, 0)  # coarsest first
    assert keys[-1] == hierarchical_tensors.max_level  # finest last
