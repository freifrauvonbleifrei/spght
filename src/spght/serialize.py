# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Binary serialization of SparseGridHierarchicalTensors, format v0.1.

Layout (little endian throughout):

File header:
    magic                  33 bytes, FormatMagic
    version                2 x uint8 (major, minor)
    num dimensions         uint16
    num subspaces          uint64
    max level              uint8 per dimension
    min level              uint8 per dimension; where the wavelet cascade
                           stops: along dimension d, a subspace with
                           level[d] == min_level[d] holds scaling
                           (approximation) coefficients, all finer levels
                           hold detail coefficients
    subspace table         one entry per subspace, in the canonical
                           (level sum, lexicographic) order:
                               level vector    uint8 per dimension
                               record offset   uint64, absolute from file start
    header checksum        uint32, crc32 of all preceding header bytes
                           (magic through subspace table), verified on read

The subspace table makes each record independently addressable.
Ordering the subspaces makes the format "scalable" in the JPEG2000 sense.

Subspace record:
    extents                uint64 per dimension
    order code             uint8: C=0, F=1, ZC=2, ZF=3
    tensor kind            uint8, TensorKind value
    value dtype            numpy kind char (1 byte) + itemsize (uint8)
    precision bits         uint16
    padding bits           uint16
    compression            uint8
    quantization_scale     float64
    quantization_offset    float64
    quantization_parameter float64
    num stored entries     uint64
    num data bytes         uint64
    checksum               uint32, crc32 of the data blob (verified on read)

    data blob:
        LINEAR             linear indices in the smallest unsigned dtype that
                           fits prod(extents), then the linear value buffer
        FULL               the linear value buffer
        EMPTY              nothing
"""

import struct
import zlib
from pathlib import Path
from typing import BinaryIO

import numpy as np

from spght.data_structures import (
    SparseGridHierarchicalTensors,
    Subspace,
    open_file,
)
from spght.linearize import Order
from spght.tensor import DenseTensor, SparseTensor, Tensor, TensorKind

FormatMagic = b"sparse grid hierarchical tensors\0"
FormatVersion: tuple[int, int] = (0, 2)

_ORDER_TO_CODE: dict[Order, int] = {"C": 0, "F": 1, "ZC": 2, "ZF": 3}
_CODE_TO_ORDER: dict[int, Order] = {c: o for o, c in _ORDER_TO_CODE.items()}

_HEADER = struct.Struct("<33sBBHQ")  # magic, major, minor, ndim, num subspaces
_HEADER_CRC = struct.Struct("<I")  # closes the header, covers all bytes before it
_RECORD = struct.Struct("<BBcBHHBdddQQI")  # see subspace record layout above


def _index_dtype(total: int) -> np.dtype:
    """Smallest little-endian unsigned dtype that can hold indices < total."""
    for dtype_str in ("<u1", "<u2", "<u4"):
        if total <= 1 << (8 * np.dtype(dtype_str).itemsize):
            return np.dtype(dtype_str)
    return np.dtype("<u8")


def _read_exactly(stream: BinaryIO, num_bytes: int) -> bytes:
    data = stream.read(num_bytes)
    if len(data) != num_bytes:
        raise ValueError(
            f"Truncated spght data: expected {num_bytes} bytes, got {len(data)}"
        )
    return data


def _open_stream(target: "str | Path | BinaryIO", mode: str) -> tuple[BinaryIO, bool]:
    if isinstance(target, (str, Path)):
        return open_file(target, mode), True
    return target, False


def _encode_record(subspace: Subspace, num_dims: int) -> bytes:
    extents = struct.pack(f"<{num_dims}Q", *subspace.extents)
    data = subspace.data
    if data is None:
        value_dtype = np.dtype(np.float64)  # placeholder, nothing is stored
        num_stored = 0
        blob = b""
    else:
        value_dtype = data.dtype
        values = data.linear_values.astype(value_dtype.newbyteorder("<"), copy=False)
        if data.kind == TensorKind.FULL:
            num_stored = data.size
            blob = values.tobytes()
        elif data.kind == TensorKind.LINEAR:
            num_stored = data.nnz
            indices = data.linear_indices.astype(_index_dtype(data.size))
            blob = indices.tobytes() + values.tobytes()
        else:
            raise ValueError(f"Cannot serialize tensor kind {data.kind!r}")
    fixed = _RECORD.pack(
        _ORDER_TO_CODE[subspace.order],
        int(subspace.kind),
        value_dtype.kind.encode("ascii"),
        value_dtype.itemsize,
        subspace.precision_bits,
        subspace.padding_bits,
        subspace.compression,
        subspace.quantization_scale,
        subspace.quantization_offset,
        subspace.quantization_parameter,
        num_stored,
        len(blob),
        zlib.crc32(blob),
    )
    return extents + fixed + blob


def _decode_record(stream: BinaryIO, num_dims: int) -> Subspace:
    extents = struct.unpack(f"<{num_dims}Q", _read_exactly(stream, 8 * num_dims))
    (
        order_code,
        kind_code,
        dtype_kind,
        itemsize,
        precision_bits,
        padding_bits,
        compression,
        quantization_scale,
        quantization_offset,
        quantization_parameter,
        num_stored,
        num_blob_bytes,
        checksum,
    ) = _RECORD.unpack(_read_exactly(stream, _RECORD.size))
    blob = _read_exactly(stream, num_blob_bytes)
    if zlib.crc32(blob) != checksum:
        raise ValueError("Subspace data block failed its checksum")

    order = _CODE_TO_ORDER[order_code]
    kind = TensorKind(kind_code)
    shape = tuple(int(e) for e in extents)
    total = int(np.prod(shape)) if shape else 1
    value_dtype = np.dtype(f"{dtype_kind.decode('ascii')}{itemsize}")

    data: Tensor | None
    if kind == TensorKind.EMPTY:
        data = None
    elif kind == TensorKind.FULL:
        if num_stored != total or num_blob_bytes != total * itemsize:
            raise ValueError("Full subspace data block has inconsistent size")
        # astype() also yields a writable copy of the read-only buffer view
        flat = np.frombuffer(blob, dtype=value_dtype.newbyteorder("<")).astype(
            value_dtype
        )
        data = DenseTensor(flat, shape, order=order)
    elif kind == TensorKind.LINEAR:
        index_dtype = _index_dtype(total)
        split = num_stored * index_dtype.itemsize
        if num_blob_bytes != split + num_stored * itemsize:
            raise ValueError("Sparse subspace data block has inconsistent size")
        keys = np.frombuffer(blob[:split], dtype=index_dtype).astype(np.int64)
        values = np.frombuffer(
            blob[split:], dtype=value_dtype.newbyteorder("<")
        ).astype(value_dtype)
        data = SparseTensor.from_linear(keys, values, shape, order=order)
    else:
        raise ValueError(f"Cannot deserialize tensor kind {kind!r}")

    return Subspace(
        extents=shape,
        precision_bits=precision_bits,
        data=data,
        quantization_scale=quantization_scale,
        quantization_offset=quantization_offset,
        quantization_parameter=quantization_parameter,
        padding_bits=padding_bits,
        compression=compression,
    )


def write(
    tensors: SparseGridHierarchicalTensors, target: "str | Path | BinaryIO"
) -> None:
    """Write the hierarchy to a path or binary stream in the v0.1 layout."""
    num_dims = tensors.dimensions
    if not 1 <= num_dims <= 65535:
        raise ValueError(f"Number of dimensions must fit in uint16, got {num_dims}")
    for level in (*tensors.subspaces.keys(), tensors.max_level, tensors.min_level):
        if any(not 0 <= single_level <= 255 for single_level in level):
            raise ValueError(f"Levels must fit in one byte each, got {level}")

    records = [
        _encode_record(subspace, num_dims) for subspace in tensors.subspaces.values()
    ]
    table_entry = struct.Struct(f"<{num_dims}BQ")
    header_size = (
        _HEADER.size
        + 2 * num_dims  # max level + min level
        + len(records) * table_entry.size
        + _HEADER_CRC.size
    )

    table = b""
    offset = header_size
    for level, record in zip(tensors.subspaces.keys(), records):
        table += table_entry.pack(*level, offset)
        offset += len(record)

    header = (
        _HEADER.pack(FormatMagic, *FormatVersion, num_dims, len(records))
        + struct.pack(f"<{num_dims}B", *tensors.max_level)
        + struct.pack(f"<{num_dims}B", *tensors.min_level)
        + table
    )
    stream, should_close = _open_stream(target, "wb")
    try:
        stream.write(header)
        stream.write(_HEADER_CRC.pack(zlib.crc32(header)))
        for record in records:
            stream.write(record)
    finally:
        if should_close:
            stream.close()


def read(source: "str | Path | BinaryIO") -> SparseGridHierarchicalTensors:
    """Read a hierarchy from a path or (seekable) binary stream."""
    stream, should_close = _open_stream(source, "rb")
    try:
        fixed_header = _read_exactly(stream, _HEADER.size)
        magic, major, minor, num_dims, num_subspaces = _HEADER.unpack(fixed_header)
        if magic != FormatMagic:
            raise ValueError("Not a spght file (magic string mismatch)")
        if (major, minor) != FormatVersion:
            raise ValueError(
                f"Unsupported format version {major}.{minor}, "
                f"expected {FormatVersion[0]}.{FormatVersion[1]}"
            )

        table_entry = struct.Struct(f"<{num_dims}BQ")
        header_rest = _read_exactly(
            stream, 2 * num_dims + num_subspaces * table_entry.size
        )
        (header_crc,) = _HEADER_CRC.unpack(_read_exactly(stream, _HEADER_CRC.size))
        if zlib.crc32(fixed_header + header_rest) != header_crc:
            raise ValueError("File header failed its checksum")

        max_level = struct.unpack(f"<{num_dims}B", header_rest[:num_dims])
        min_level = struct.unpack(
            f"<{num_dims}B", header_rest[num_dims : 2 * num_dims]
        )
        table: list[tuple[tuple[int, ...], int]] = []
        for i in range(num_subspaces):
            entry = table_entry.unpack_from(
                header_rest, 2 * num_dims + i * table_entry.size
            )
            table.append((entry[:-1], entry[-1]))

        subspaces: dict[tuple[int, ...], Subspace] = {}
        for level, offset in table:
            stream.seek(offset)
            subspaces[level] = _decode_record(stream, num_dims)
    finally:
        if should_close:
            stream.close()

    return SparseGridHierarchicalTensors(
        dimensions=num_dims,
        max_level=tuple(max_level),
        min_level=tuple(min_level),
        subspaces=subspaces,
    )
