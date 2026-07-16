# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from spght.data_structures import (
    Convention,
    NumberFormat,
    SparseGridHierarchicalTensors,
    Subspace,
    TensorKind,
    subspace_order_key,
)
from spght.hierarchize import hierarchize
from spght.tensor import DenseTensor, IntervalTensor, SparseTensor, make_tensor


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
    assert sparse.kind == TensorKind.POINTWISE and sparse.is_sparse
    # the kind is a class-level property of each implementation
    assert DenseTensor.kind == TensorKind.FULL
    assert SparseTensor.kind == TensorKind.POINTWISE
    # numeric values are the (future) on-disk index-kind identifiers
    assert int(TensorKind.EMPTY) == 0
    assert int(TensorKind.FULL) == 1
    assert int(TensorKind.POINTWISE) == 2
    assert int(TensorKind.INTERVALS) == 3
    assert IntervalTensor.kind == TensorKind.INTERVALS
    assert IntervalTensor.from_dense(array).is_sparse
    # a subspace without data defaults to the EMPTY kind
    assert Subspace(extents=(2, 2), precision_bits=64).kind == TensorKind.EMPTY


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_interval_from_dense_roundtrip(order):
    array = np.zeros((4, 8))
    array[1, 2:7] = np.arange(1.0, 6.0)  # one row-run
    array[3, 0] = 7.0  # an isolated point
    t = IntervalTensor.from_dense(array, order=order)
    assert t.is_sparse
    assert t.nnz == 6
    assert np.array_equal(t.to_dense(), array)
    # semantic equality across storage kinds, per Tensor.__eq__
    assert t == SparseTensor.from_dense(array, order=order)
    assert t == DenseTensor.from_dense(array, order=order)
    # the covered set matches the pointwise one exactly
    assert np.array_equal(
        t.linear_indices, SparseTensor.from_dense(array, order=order).linear_indices
    )


def test_interval_run_coalescing():
    keys = np.array([12, 3, 4, 5, 9, 13])  # unsorted on purpose
    values = np.arange(6, dtype=np.float64)
    t = IntervalTensor.from_linear(keys, values, shape=(16,))
    assert t.num_runs == 3  # (3..5), (9), (12..13)
    assert np.array_equal(t.linear_indices, [3, 4, 5, 9, 12, 13])
    # values follow their sorted keys
    assert t[np.array([3, 9, 12])].tolist() == [1.0, 4.0, 0.0]


def test_interval_get_set_semantics():
    t = IntervalTensor(
        firsts=[2, 8], lasts=[4, 9], values=np.arange(5.0), shape=(4, 4), order="C"
    )
    assert t[3] == 1.0  # covered linear index
    assert t[0, 2] == 0.0  # multidim (0,2) -> linear 2 -> covered
    assert t[5] == 0.0  # implicit zero between the runs
    t[8] = -1.0
    assert t[2, 0] == -1.0
    t[3] = 0.0  # zero write inside a run keeps the run structure
    assert t.num_runs == 2 and t.nnz == 5
    with pytest.raises(NotImplementedError, match="runs"):
        t[5] = 1.0  # cannot insert outside the covered set
    with pytest.raises(IndexError):
        t[16]


def test_interval_validation():
    values3 = np.ones(3)
    with pytest.raises(ValueError, match="within"):
        IntervalTensor(firsts=[14], lasts=[16], values=values3, shape=(4, 4))
    with pytest.raises(ValueError, match="last >= first"):
        IntervalTensor(firsts=[3], lasts=[1], values=values3, shape=(4, 4))
    with pytest.raises(ValueError, match="disjoint"):
        IntervalTensor(firsts=[0, 2], lasts=[3, 5], values=np.ones(8), shape=(4, 4))
    with pytest.raises(ValueError, match="maximal"):
        # adjacent runs must be merged into one
        IntervalTensor(firsts=[0, 3], lasts=[2, 5], values=np.ones(6), shape=(4, 4))
    with pytest.raises(ValueError, match="values"):
        IntervalTensor(firsts=[0], lasts=[2], values=np.ones(2), shape=(4, 4))
    with pytest.raises(ValueError, match="unique"):
        IntervalTensor.from_linear([1, 1], np.ones(2), shape=(4, 4))


@pytest.mark.parametrize("order", ["C", "F", "ZC", "ZF"])
def test_interval_with_order(order):
    array = np.zeros((4, 8))
    array[2, 1:6] = 3.0
    t = IntervalTensor.from_dense(array, order="C")
    relinearized = t.with_order(order)
    assert isinstance(relinearized, IntervalTensor)
    assert relinearized == IntervalTensor.from_dense(array, order=order)
    if order == "C":
        assert relinearized is t


def test_interval_nbytes_and_items():
    array = np.zeros((4, 4))
    array[0, :2] = [1.0, 2.0]
    t = IntervalTensor.from_dense(array, order="C")
    assert t.num_runs == 1
    assert t.nbytes == 2 * 8 + 2 * 8  # two int64 bounds + two float64 values
    assert list(t.nonzero_items()) == [((0, 0), 1.0), ((0, 1), 2.0)]
    # explicit zeros inside a run are skipped by nonzero_items
    t[0] = 0.0
    assert list(t.nonzero_items()) == [((0, 1), 2.0)]


def test_interval_empty():
    t = IntervalTensor.from_dense(np.zeros((4, 4)), order="C")
    assert t.nnz == 0 and t.num_runs == 0
    assert t[7] == 0.0
    assert np.array_equal(t.to_dense(), np.zeros((4, 4)))
    assert t.linear_indices.size == 0


def test_subspace_holds_tensor():
    array = np.arange(8, dtype=np.float64).reshape(2, 4)
    dense = DenseTensor.from_dense(array, order="C")
    subspace = Subspace(extents=(2, 4), precision_bits=64, data=dense)
    assert subspace.kind == TensorKind.FULL
    assert not subspace.is_sparse
    assert subspace.num_bytes == dense.nbytes
    assert np.array_equal(subspace.data, array)

    sparse_sub = Subspace(
        extents=(2, 4),
        precision_bits=64,
        data=SparseTensor.from_dense(array, order="C"),
    )
    assert sparse_sub.kind == TensorKind.POINTWISE
    assert sparse_sub.is_sparse

    with pytest.raises(ValueError):
        Subspace(extents=(4, 2), precision_bits=64, data=dense)  # shape mismatch


def test_subspace_validates_num_components():
    for out_of_byte_range in (0, -1, 256):
        with pytest.raises(ValueError, match="num_components"):
            Subspace(extents=(2,), precision_bits=64, num_components=out_of_byte_range)
    subspace = Subspace(extents=(2,), precision_bits=64)
    assert subspace.num_components == 1
    assert subspace.component_layout == 0
    with pytest.raises(ValueError, match="component_layout"):
        Subspace(extents=(2,), precision_bits=64, component_layout=2)


def test_number_format_byte_roundtrip():
    named_formats = [
        NumberFormat(exponent_bits=11),  # float64
        NumberFormat(exponent_bits=8),  # float32 / bfloat16 / tf32
        NumberFormat(exponent_bits=5),  # float16 / fp8 E5M2
        NumberFormat(exponent_bits=4, convention=Convention.FN),  # fp8 E4M3FN
        NumberFormat(exponent_bits=5, convention=Convention.FNUZ),
        NumberFormat(exponent_bits=2),  # fp4 E2M1
        NumberFormat(exponent_bits=8, unsigned=True, convention=Convention.FN),
        NumberFormat(),  # signed integer
        NumberFormat(unsigned=True),  # unsigned integer
        NumberFormat(convention=Convention.OTHER),
    ]
    seen_bytes = set()
    for number_format in named_formats:
        byte = number_format.to_byte()
        assert 0 <= byte <= 255
        assert NumberFormat.from_byte(byte) == number_format
        seen_bytes.add(byte)
    assert len(seen_bytes) == len(named_formats)  # encoding is injective


def test_number_format_from_dtype():
    assert NumberFormat.from_dtype(np.float64) == NumberFormat(exponent_bits=11)
    assert NumberFormat.from_dtype(np.float32) == NumberFormat(exponent_bits=8)
    assert NumberFormat.from_dtype(np.float16) == NumberFormat(exponent_bits=5)
    assert NumberFormat.from_dtype(np.int8) == NumberFormat()
    assert NumberFormat.from_dtype(np.uint16) == NumberFormat(unsigned=True)


def test_number_format_validation():
    with pytest.raises(ValueError, match="exponent_bits"):
        NumberFormat(exponent_bits=32)  # does not fit the 5-bit field
    with pytest.raises(ValueError, match="FN"):
        NumberFormat(convention=Convention.FN)  # floats need an exponent
    with pytest.raises(ValueError, match="OTHER"):
        NumberFormat(exponent_bits=4, convention=Convention.OTHER)
    with pytest.raises(ValueError, match="OTHER"):
        NumberFormat(unsigned=True, convention=Convention.OTHER)


def test_subspace_number_format():
    # derived from the data dtype when not given
    data = DenseTensor.from_dense(np.zeros((2,), dtype=np.float32))
    subspace = Subspace(extents=(2,), precision_bits=32, data=data)
    assert subspace.number_format == NumberFormat(exponent_bits=8)
    # without data: float64's flavor
    assert Subspace(extents=(2,), precision_bits=64).number_format == (
        NumberFormat(exponent_bits=11)
    )
    # sign + exponent bits must fit in precision_bits: float64's flavor
    # needs 12 of them
    with pytest.raises(ValueError, match="precision_bits is 8"):
        Subspace(
            extents=(2,),
            precision_bits=8,
            number_format=NumberFormat(exponent_bits=11),
        )


def test_hierarchize_produces_dense_tensors():
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
    nodal_values = np.random.default_rng(4).random((4, 2, 4))
    hierarchical_tensors = hierarchize(nodal_values)
    keys = list(hierarchical_tensors.subspaces.keys())
    assert keys == sorted(keys, key=subspace_order_key)
    assert keys[0] == (0, 0, 0)  # coarsest first
    assert keys[-1] == hierarchical_tensors.max_level  # finest last


def test_min_level_and_scaling_dimensions():
    tensors = SparseGridHierarchicalTensors(
        dimensions=2, max_level=(3, 3), min_level=(1, 2)
    )
    # scaling exactly where the level equals min_level
    assert tensors.scaling_dimensions((1, 2)) == (True, True)
    assert tensors.scaling_dimensions((1, 3)) == (True, False)
    assert tensors.scaling_dimensions((3, 2)) == (False, True)
    assert tensors.scaling_dimensions((3, 3)) == (False, False)

    # default is all zeros
    assert SparseGridHierarchicalTensors(
        dimensions=3, max_level=(1, 1, 1)
    ).min_level == (0, 0, 0)

    with pytest.raises(ValueError):  # min_level must not exceed max_level
        SparseGridHierarchicalTensors(dimensions=2, max_level=(1, 1), min_level=(2, 0))
    with pytest.raises(ValueError):  # subspace keys must respect min_level
        SparseGridHierarchicalTensors(
            dimensions=2,
            max_level=(2, 2),
            min_level=(1, 1),
            subspaces={
                (0, 1): Subspace(
                    extents=(1, 1),
                    precision_bits=64,
                    data=DenseTensor.from_dense(np.ones((1, 1))),
                )
            },
        )


def test_sparse_from_linear_validates_keys():
    values = np.array([1.0, 2.0])
    # out-of-range keys would otherwise silently wrap in the narrowing
    # index cast at serialization time
    with pytest.raises(ValueError, match="within"):
        SparseTensor.from_linear(np.array([0, 300]), values, shape=(16, 16))
    with pytest.raises(ValueError, match="within"):
        SparseTensor.from_linear(np.array([-1, 3]), values, shape=(16, 16))
    with pytest.raises(ValueError, match="unique"):
        SparseTensor.from_linear(np.array([5, 5]), values, shape=(16, 16))
    # valid keys still work, unsorted input included
    tensor = SparseTensor.from_linear(np.array([7, 3]), values, shape=(16, 16))
    assert tensor.nnz == 2 and tensor[3] == 2.0


def test_tensor_semantic_equality():
    array = np.zeros((4, 4))
    array[1, 2] = 3.0
    dense = DenseTensor.from_dense(array)
    sparse = SparseTensor.from_dense(array)
    # equality is semantic: storage kind does not matter
    assert dense == sparse
    assert sparse == DenseTensor.from_dense(array)
    assert dense != DenseTensor.from_dense(array, order="F")  # order differs
    assert dense != DenseTensor.from_dense(array.astype(np.float32))
    other = array.copy()
    other[0, 0] = 1.0
    assert dense != DenseTensor.from_dense(other)
    # mutable -> unhashable
    with pytest.raises(TypeError):
        hash(dense)


def test_add_subspace_validates_like_the_constructor():
    tensors = SparseGridHierarchicalTensors(dimensions=2, max_level=(1, 1))
    with pytest.raises(ValueError, match="extents dimensionality"):
        tensors.add_subspace(
            (1, 1),
            Subspace(
                extents=(1,),  # wrong dimensionality
                precision_bits=64,
                data=DenseTensor.from_dense(np.ones(1)),
            ),
        )
