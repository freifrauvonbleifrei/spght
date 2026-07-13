import io

import numpy as np
import pytest

from spght.compress import compress
from spght.data_structures import (
    SparseGridHierarchicalTensors,
    Subspace,
    TensorKind,
    subspace_order_key,
)
from spght.hierarchize import hierarchize
from spght.serialize import _index_dtype
from spght.tensor import DenseTensor, SparseTensor


def _assert_equal_containers(
    expected: SparseGridHierarchicalTensors, actual: SparseGridHierarchicalTensors
) -> None:
    assert actual.dimensions == expected.dimensions
    assert actual.max_level == tuple(int(level) for level in expected.max_level)
    assert list(actual.subspaces.keys()) == list(expected.subspaces.keys())
    for level, expected_subspace in expected.subspaces.items():
        actual_subspace = actual.subspaces[level]
        assert actual_subspace.extents == tuple(expected_subspace.extents)
        assert actual_subspace.precision_bits == expected_subspace.precision_bits
        assert actual_subspace.padding_bits == expected_subspace.padding_bits
        assert actual_subspace.compression == expected_subspace.compression
        assert actual_subspace.quantization_scale == expected_subspace.quantization_scale
        assert actual_subspace.quantization_offset == expected_subspace.quantization_offset
        assert (
            actual_subspace.quantization_parameter
            == expected_subspace.quantization_parameter
        )
        assert actual_subspace.kind == expected_subspace.kind
        assert actual_subspace.order == expected_subspace.order
        if expected_subspace.data is None:
            assert actual_subspace.data is None
        else:
            assert actual_subspace.data is not None
            assert actual_subspace.data.dtype == expected_subspace.data.dtype
            # bit-exact round-trip of the linear storage
            assert np.array_equal(
                actual_subspace.data.linear_indices,
                expected_subspace.data.linear_indices,
            )
            assert np.array_equal(
                actual_subspace.data.linear_values,
                expected_subspace.data.linear_values,
            )


def test_roundtrip_hierarchized_via_path(tmp_path):
    nodal_values = np.random.default_rng(0).random((4, 8))
    tensors = hierarchize(nodal_values)
    path = tmp_path / "hierarchized.spght"
    tensors.write(path)
    _assert_equal_containers(tensors, SparseGridHierarchicalTensors.read(path))
    # str paths work too
    _assert_equal_containers(tensors, SparseGridHierarchicalTensors.read(str(path)))


def test_roundtrip_compressed_via_bytesio():
    # localized bump -> compression produces a mix of FULL and LINEAR kinds
    x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    nodal_values = np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.002)
    tensors = compress(hierarchize(nodal_values), epsilon=1e-3)
    kinds = {subspace.kind for subspace in tensors.subspaces.values()}
    assert kinds == {TensorKind.FULL, TensorKind.LINEAR}  # exercises both

    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    _assert_equal_containers(tensors, SparseGridHierarchicalTensors.read(buffer))


def test_roundtrip_all_kinds_orders_and_dtypes(tmp_path):
    rng = np.random.default_rng(1)
    sparse_values = np.zeros((4, 4), dtype=np.float64)
    sparse_values[1, 2] = 3.0
    sparse_values[3, 0] = -4.0
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(2, 2),
        subspaces={
            (0, 0): Subspace(
                extents=(1, 1),
                precision_bits=64,
                data=DenseTensor.from_dense(np.array([[2.5]])),
            ),
            (1, 0): Subspace(extents=(1, 1), precision_bits=64, data=None),  # EMPTY
            (1, 2): Subspace(
                extents=(2, 4),
                precision_bits=32,
                data=DenseTensor.from_dense(
                    rng.random((2, 4)).astype(np.float32), order="ZF"
                ),
            ),
            (2, 2): Subspace(
                extents=(4, 4),
                precision_bits=64,
                data=SparseTensor.from_dense(sparse_values, order="ZC"),
                padding_bits=3,
                compression=0,
            ),
            (2, 1): Subspace(
                extents=(2, 2),
                precision_bits=8,
                # int8-stored coefficients + the reserved quantization fields
                data=DenseTensor.from_dense(
                    np.array([[1, 2], [3, 4]], dtype=np.int8)
                ),
                quantization_scale=0.25,
                quantization_offset=2.0,
                quantization_parameter=-1.5,
            ),
        },
    )
    path = tmp_path / "mixed.spght"
    tensors.write(path)
    read_back = SparseGridHierarchicalTensors.read(path)
    _assert_equal_containers(tensors, read_back)
    assert read_back.subspaces[(1, 0)].kind == TensorKind.EMPTY
    assert read_back.subspaces[(1, 2)].data.dtype == np.float32
    assert read_back.subspaces[(2, 2)].order == "ZC"
    quantized = read_back.subspaces[(2, 1)]
    assert quantized.data.dtype == np.int8
    assert (
        quantized.quantization_scale,
        quantized.quantization_offset,
        quantized.quantization_parameter,
    ) == (0.25, 2.0, -1.5)


def test_roundtrip_empty_container():
    tensors = SparseGridHierarchicalTensors(dimensions=3, max_level=(0, 0, 0))
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    _assert_equal_containers(tensors, SparseGridHierarchicalTensors.read(buffer))


def test_file_preserves_canonical_subspace_order():
    tensors = hierarchize(np.random.default_rng(2).random((4, 4)))
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    keys = list(SparseGridHierarchicalTensors.read(buffer).subspaces.keys())
    assert keys == sorted(keys, key=subspace_order_key)


def _valid_file_bytes() -> bytearray:
    tensors = hierarchize(np.random.default_rng(3).random((4, 4)))
    buffer = io.BytesIO()
    tensors.write(buffer)
    return bytearray(buffer.getvalue())


def test_read_rejects_bad_magic():
    corrupted = _valid_file_bytes()
    corrupted[0] ^= 0xFF
    with pytest.raises(ValueError, match="magic"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_rejects_unsupported_version():
    corrupted = _valid_file_bytes()
    corrupted[33] = 9  # major version byte follows the 33-byte magic
    with pytest.raises(ValueError, match="version"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_detects_corrupted_header():
    corrupted = _valid_file_bytes()
    # byte 45 is the first max_level byte, i.e. inside the checksummed
    # header but past the magic/version fields that are checked first
    corrupted[45] ^= 0xFF
    with pytest.raises(ValueError, match="header"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_many_dimensions_roundtrip():
    # the format allows up to 65535 dimensions (uint16); numpy arrays cap at
    # 64 axes, so an EMPTY subspace exercises a 300-dimensional file
    dims = 300
    tensors = SparseGridHierarchicalTensors(
        dimensions=dims,
        max_level=(0,) * dims,
        subspaces={(0,) * dims: Subspace(extents=(1,) * dims, precision_bits=64)},
    )
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    _assert_equal_containers(tensors, SparseGridHierarchicalTensors.read(buffer))


def test_read_detects_corrupted_data():
    corrupted = _valid_file_bytes()
    corrupted[-1] ^= 0xFF  # last byte belongs to the last subspace's values
    with pytest.raises(ValueError, match="checksum"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_rejects_truncated_file():
    truncated = _valid_file_bytes()[:-10]
    with pytest.raises(ValueError, match="[Tt]runcated"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(truncated)))


def test_index_dtype_is_minimal():
    assert _index_dtype(256) == np.dtype("<u1")  # indices go up to 255
    assert _index_dtype(257) == np.dtype("<u2")
    assert _index_dtype(1 << 16) == np.dtype("<u2")
    assert _index_dtype((1 << 16) + 1) == np.dtype("<u4")
    assert _index_dtype(1 << 32) == np.dtype("<u4")
    assert _index_dtype((1 << 32) + 1) == np.dtype("<u8")
