# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0


from dataclasses import replace
import io
import numpy as np
import pytest
import struct as _struct
import zlib as _zlib

from spght.basis import Dirichlet, Neumann
from spght.compress import compress
from spght.data_structures import (
    SparseGridHierarchicalTensors,
    Subspace,
    TensorKind,
    subspace_order_key,
)
from spght.hierarchize import hierarchize, dehierarchize
from spght.lifting import Basis1D
from spght.serialize import _index_dtype
from spght.tensor import DenseTensor, SparseTensor
from spght.wavelets import haar_basis, cdf_2_2_basis, cubic_basis, hat_basis


def _assert_equal_containers(
    expected: SparseGridHierarchicalTensors, actual: SparseGridHierarchicalTensors
) -> None:
    assert actual.dimensions == expected.dimensions
    assert actual.max_level == tuple(int(level) for level in expected.max_level)
    assert actual.min_level == tuple(int(level) for level in expected.min_level)
    assert actual.bases == expected.bases  # structural Basis1D equality
    assert list(actual.subspaces.keys()) == list(expected.subspaces.keys())
    for level, expected_subspace in expected.subspaces.items():
        actual_subspace = actual.subspaces[level]
        assert actual_subspace.extents == tuple(expected_subspace.extents)
        assert actual_subspace.precision_bits == expected_subspace.precision_bits
        assert actual_subspace.padding_bits == expected_subspace.padding_bits
        assert actual_subspace.compression == expected_subspace.compression
        assert (
            actual_subspace.quantization_scale == expected_subspace.quantization_scale
        )
        assert (
            actual_subspace.quantization_zero_point
            == expected_subspace.quantization_zero_point
        )
        assert actual_subspace.order == expected_subspace.order
        if expected_subspace.data is None:
            assert actual_subspace.kind == TensorKind.EMPTY
            assert actual_subspace.data is None
        else:
            assert actual_subspace.data is not None
            assert actual_subspace.data.dtype == expected_subspace.data.dtype
            # the on-disk kind is re-chosen at write time (whichever of
            # FULL/LINEAR is smaller), so compare bit-exact dense views
            assert np.array_equal(
                actual_subspace.data.to_dense(),
                expected_subspace.data.to_dense(),
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
                # int8-stored coefficients; the reserved quantization
                # fields must stay identity to be serializable
                data=DenseTensor.from_dense(np.array([[1, 2], [3, 4]], dtype=np.int8)),
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
    assert read_back.subspaces[(2, 1)].data.dtype == np.int8


def test_write_picks_cheapest_on_disk_kind():
    almost_empty = np.zeros((4, 4))
    almost_empty[2, 1] = 7.0
    full = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
    tensors = SparseGridHierarchicalTensors(
        dimensions=2,
        max_level=(2, 2),
        subspaces={
            (2, 1): Subspace(
                extents=(4, 4),
                precision_bits=64,
                data=DenseTensor.from_dense(almost_empty),
            ),
            (2, 2): Subspace(
                extents=(4, 4),
                precision_bits=64,
                data=SparseTensor.from_dense(full),
            ),
        },
    )
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    loaded = SparseGridHierarchicalTensors.read(buffer)
    # the dense-stored but nearly-empty subspace comes back sparse, the
    # sparse-stored but fully-populated one comes back dense ...
    assert loaded.subspaces[(2, 1)].kind == TensorKind.LINEAR
    assert loaded.subspaces[(2, 2)].kind == TensorKind.FULL
    # ... both with unchanged values
    for level in tensors.subspaces:
        assert np.array_equal(
            loaded.subspaces[level].data.to_dense(),
            tensors.subspaces[level].data.to_dense(),
        )


def test_roundtrip_min_level():
    nodal_values = np.random.default_rng(4).random((8, 8))
    tensors = hierarchize(nodal_values, min_level=(1, 2))
    assert tensors.min_level == (1, 2)
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    loaded = SparseGridHierarchicalTensors.read(buffer)
    _assert_equal_containers(tensors, loaded)
    # scaling-ness is derived identically on both sides
    for level in tensors.subspaces:
        assert loaded.scaling_dimensions(level) == tensors.scaling_dimensions(level)


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


def test_roundtrip_basis_descriptors():
    rng = np.random.default_rng(13)
    # anisotropic bases with a boundary rule that carries data (Dirichlet
    # wall value) -> exercises the non-uniform basis block
    bases = (
        hat_basis(bc_left=Dirichlet(1.5), bc_right=Neumann()),
        cdf_2_2_basis(),
    )
    nodal_values = rng.normal(size=(9, 17))  # vertex-centered: 2^l + 1
    tensors = hierarchize(nodal_values, wavelet=bases)

    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    loaded = SparseGridHierarchicalTensors.read(buffer)
    _assert_equal_containers(tensors, loaded)
    assert loaded.bases == bases

    # the loaded container reconstructs WITHOUT the basis being passed again
    assert np.allclose(dehierarchize(loaded), nodal_values, atol=1e-11)


def test_roundtrip_uniform_basis_block_is_shared():
    rng = np.random.default_rng(14)
    basis = cubic_basis()
    tensors = hierarchize(rng.normal(size=(17, 17)), wavelet=basis)
    uniform_buffer = io.BytesIO()
    tensors.write(uniform_buffer)

    # same content with explicitly repeated (equal) bases must produce an
    # identical file: the writer detects uniformity structurally
    tensors_repeated = hierarchize(rng.normal(size=(17, 17)), wavelet=(basis, basis))
    assert tensors_repeated.bases == tensors.bases

    loaded = SparseGridHierarchicalTensors.read(io.BytesIO(uniform_buffer.getvalue()))
    assert loaded.bases == (basis, basis)


def test_periodic_basis_roundtrip():
    rng = np.random.default_rng(15)
    basis = cdf_2_2_basis(periodic=True)
    nodal_values = rng.normal(size=16)  # periodic vertex grid: 2^l dofs
    tensors = hierarchize(nodal_values, wavelet=basis)
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    loaded = SparseGridHierarchicalTensors.read(buffer)
    assert loaded.bases == (basis,)
    assert np.allclose(dehierarchize(loaded), nodal_values, atol=1e-11)


def test_scheme_name_limited_to_15_characters():
    basis = haar_basis()
    too_long = Basis1D(
        basis.centering,
        replace(basis.scheme, name="a" * 16),
        basis.bc_left,
        basis.bc_right,
    )
    tensors = hierarchize(np.zeros(4), wavelet=too_long)
    with pytest.raises(ValueError, match="15 ascii characters"):
        tensors.write(io.BytesIO())
    # 15 characters are fine and round-trip
    just_fits = Basis1D(
        basis.centering,
        replace(basis.scheme, name="a" * 15),
        basis.bc_left,
        basis.bc_right,
    )
    tensors = hierarchize(np.zeros(4), wavelet=just_fits)
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    assert SparseGridHierarchicalTensors.read(buffer).bases[0].scheme.name == "a" * 15


def _first_record_offset(raw: bytes) -> tuple[int, int]:
    (num_dims,) = _struct.unpack_from("<H", raw, 35)
    (num_subspaces,) = _struct.unpack_from("<Q", raw, 37)
    (basis_length,) = _struct.unpack_from("<I", raw, 45 + 2 * num_dims)
    return (
        45 + 2 * num_dims + 4 + basis_length + num_subspaces * (num_dims + 8) + 4
    ), num_dims


def test_read_rejects_unknown_order_code():
    corrupted = _valid_file_bytes()
    record, num_dims = _first_record_offset(corrupted)
    corrupted[record + 4 * num_dims] = 9  # order code, outside the record CRC
    with pytest.raises(ValueError, match="order code"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_rejects_reserved_compression_byte():
    corrupted = _valid_file_bytes()
    record, num_dims = _first_record_offset(corrupted)
    # after order, kind, dtype char, itemsize, precision (2), padding (2)
    corrupted[record + 8 * num_dims + 8] = 5
    with pytest.raises(ValueError, match="compression"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_rejects_invalid_dtype():
    corrupted = _valid_file_bytes()
    record, num_dims = _first_record_offset(corrupted)
    corrupted[record + 8 * num_dims + 2] = ord("x")  # dtype kind char
    with pytest.raises(ValueError, match="dtype"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_read_rejects_duplicate_table_entries():

    corrupted = _valid_file_bytes()
    (num_dims,) = _struct.unpack_from("<H", corrupted, 35)
    (num_subspaces,) = _struct.unpack_from("<Q", corrupted, 37)
    assert num_subspaces >= 2
    (basis_length,) = _struct.unpack_from("<I", corrupted, 45 + 2 * num_dims)
    table = 45 + 2 * num_dims + 4 + basis_length
    entry_size = num_dims + 8
    # copy the first table entry over the second, then re-seal the header CRC
    corrupted[table + entry_size : table + 2 * entry_size] = corrupted[
        table : table + entry_size
    ]
    crc_offset = table + num_subspaces * entry_size
    _struct.pack_into("<I", corrupted, crc_offset, _zlib.crc32(corrupted[:crc_offset]))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_container_equality_roundtrip():
    x, y = np.meshgrid(np.linspace(0, 1, 32), np.linspace(0, 1, 32), indexing="ij")
    nodal_values = np.exp(-((x - 0.3) ** 2 + (y - 0.6) ** 2) / 0.002)
    tensors = compress(hierarchize(nodal_values), epsilon=1e-3)
    buffer = io.BytesIO()
    tensors.write(buffer)
    buffer.seek(0)
    assert SparseGridHierarchicalTensors.read(buffer) == tensors

def test_read_rejects_nonidentity_quantization_fields():
    corrupted = _valid_file_bytes()
    record, num_dims = _first_record_offset(corrupted)
    # the quantization scale sits after order(1) + kind(1) + dtype char(1)
    # + itemsize(1) + precision(2) + padding(2) + compression(1) = 9 bytes
    # of the fixed record part, which is not covered by the blob CRC
    _struct.pack_into("<f", corrupted, record + 4 * num_dims + 9, 2.0)
    with pytest.raises(ValueError, match="quantization"):
        SparseGridHierarchicalTensors.read(io.BytesIO(bytes(corrupted)))


def test_write_rejects_extents_beyond_uint32():
    oversized = Subspace(extents=(2**33, 16), precision_bits=64)  # EMPTY data
    tensors = SparseGridHierarchicalTensors(
        dimensions=2, max_level=(1, 1), subspaces={(1, 1): oversized}
    )
    with pytest.raises(ValueError, match="uint32"):
        tensors.write(io.BytesIO())
