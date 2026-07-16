# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Binary serialization of SparseGridHierarchicalTensors, format v0.5.

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
    basis block            uint32 byte length (so readers can skip the block
                           without parsing it), then:
                               uniformity      uint8: 1 = one basis descriptor
                                               shared by all dimensions,
                                               0 = one descriptor per dimension
                               descriptor(s)   see basis descriptor below
    metadata block         uint32 byte length (skippable wholesale), then:
                               num entries     uint16
                               entries         see metadata entry below
    subspace table         one entry per subspace, in the canonical
                           (level sum, lexicographic) order:
                               level vector    uint8 per dimension
                               record offset   uint64, relative to the start
                                               of the spght payload (the
                                               first magic byte) -- equal to
                                               an absolute file offset when
                                               the payload starts at byte 0
    header checksum        uint32, crc32 of all preceding header bytes
                           (magic through subspace table), verified on read

Metadata entry -- descriptive key-value payload. Metadata NEVER influences
how the rest of the file is parsed; a reader that skips the whole block
loses labels, not correctness:
    key length             uint8, 1..63
    key                    ascii, charset [a-z0-9_.-]; keys are unique;
                           un-prefixed keys are reserved for this spec
                           (suggested: field_name, dimension_names,
                           domain_min, domain_max, time, created_by),
                           applications should prefix theirs (e.g. "x-")
    value tag              uint8: utf-8 string=0, float64=1, int64=2,
                           float64 array=3, int64 array=4, bytes=5,
                           string array=6 (uint16 count, then per string
                           uint16 byte length + utf-8 bytes)
    payload length         uint32, validated against the enclosing block
                           before any read
    payload                per-tag encoding; readers preserve entries with
                           unknown tags verbatim (additive extensibility)
                           and never interpret them

Basis descriptor -- the lifting program of one dimension's wavelet, fully
self-describing (a reader reconstructs by running the steps backwards):
    centering              uint8: cell=0, vertex=1, vertex-interior=2,
                           vertex-periodic=3
    boundary rule left     uint8 (Extrapolate=0, Neumann=1, Periodic=2,
                           Dirichlet=3) + float64 wall value (Dirichlet only,
                           0.0 otherwise)
    boundary rule right    uint8 + float64, as above
    evaluation hint        uint8: none=0, midpoint=1, nodal_linear=2
    scheme name            16 bytes: zero-terminated ascii, c-style (at most
                           15 characters, padded with NUL bytes; a label,
                           never semantic)
    num lifting steps      uint8, then per step:
                               kind            uint8: predict=0, update=1,
                                               scale_detail=2
                               factor          float64
                               num taps        uint8, then per tap:
                                                   offset  int8
                                                   weight  float64

The subspace table makes each record independently addressable.
Ordering the subspaces makes the format "scalable" in the JPEG2000 sense.

Subspace record:
    extents                uint32 per dimension
    order code             uint8: C=0, F=1, ZC=2, ZF=3
    tensor kind            uint8, TensorKind value
    value dtype            numpy kind char (1 byte) + itemsize (uint8);
    num components         uint8; reserved (must be 1): number of values
                           stored per spatial point.
    component layout       uint8; reserved (must be 0): how the values of
                           a multi-component blob are arranged --
                           planes=0 (SoA, one complete spatial
                           linearization per component, back to back) or
                           interleaved=1 (AoS, the components of each
                           point stored together, points in linearization
                           order).
    precision bits         uint8, 1 to 8 * dtype itemsize: total bits of
                           the stored scalars' number format
    number format          uint8, the format's flavor: bits 0-4 exponent
                           width e (0 = integer), bit 5 unsigned flag,
                           bits 6-7 special-value convention (IEEE=0,
                           FN=1, FNUZ=2, OTHER=3). The bias is
                           implied by the convention (IEEE/FN:
                           2^(e-1)-1, FNUZ: 2^(e-1)). The container
                           dtype holds the exact widened values, so
                           readers decode correctly without interpreting
                           this byte
    padding bits           uint16; trailing slack bits, meaningful only
                           under a future bit-packing transform
                           (carried verbatim until then)
    compression            uint8
    quantization_scale     float32; reserved, with the future semantics
    quantization_zero_pt   int32     logical = scale * (stored - zero_point)
    num stored entries     uint64
    num data bytes         uint64
    data blob:
        POINTWISE          linear indices in the smallest unsigned dtype that
                           fits prod(extents), then the linear value buffer
        INTERVALS          run bounds in that same index dtype -- all run
                           firsts, then the matching run lasts (both
                           inclusive) -- then the values of every covered
                           position, run by run. Runs are sorted, disjoint,
                           and separated by >= 1 uncovered position; the
                           run count is derived from the blob size. Covered
                           values may include explicit zeros: the writer
                           bridges zero gaps into a run while that is no
                           costlier than an extra pair of bounds
        FULL               the linear value buffer
        EMPTY              nothing
    checksum               uint32, crc32 of the data blob (verified on read);
                           placed after the blob so a writer can stream the
                           blob while accumulating the checksum
"""

import math
import struct
import zlib
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np

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
from spght.data_structures import (
    MetadataValue,
    NumberFormat,
    OpaqueValue,
    SparseGridHierarchicalTensors,
    Subspace,
)
from spght.lifting import Basis1D, LiftingScheme, LiftingStep
from spght.linearize import Order
from spght.tensor import (
    DenseTensor,
    IntervalTensor,
    SparseTensor,
    Tensor,
    TensorKind,
    runs_from_sorted_keys,
)
from spght.util import per_dimension

FormatMagic = b"sparse grid hierarchical tensors\0"
FormatVersion: tuple[int, int] = (0, 5)

_ORDER_TO_CODE: dict[Order, int] = {"C": 0, "F": 1, "ZC": 2, "ZF": 3}
_CODE_TO_ORDER: dict[int, Order] = {c: o for o, c in _ORDER_TO_CODE.items()}

_HEADER = struct.Struct("<33sBBHQ")  # magic, major, minor, ndim, num subspaces
_HEADER_CRC = struct.Struct("<I")  # closes the header, covers all bytes before it
_RECORD = struct.Struct("<BBcBBBBBHBfiQQ")  # see subspace record layout above
_RECORD_CRC = struct.Struct("<I")  # follows each record's data blob


_BC = struct.Struct("<Bd")  # boundary rule code + Dirichlet wall value
_STEP = struct.Struct("<BdB")  # step kind, factor, num taps
_TAP = struct.Struct("<bd")  # stencil offset, weight

_BC_TYPE_TO_CODE: dict[type, int] = {
    Extrapolate: 0,
    Neumann: 1,
    Periodic: 2,
    Dirichlet: 3,
}
_EVALUATION_TO_CODE = {None: 0, "midpoint": 1, "nodal_linear": 2}
_CODE_TO_EVALUATION = {code: name for name, code in _EVALUATION_TO_CODE.items()}
_STEP_KIND_TO_CODE = {"predict": 0, "update": 1, "scale_detail": 2}
_CODE_TO_STEP_KIND = {code: name for name, code in _STEP_KIND_TO_CODE.items()}


def _centering_code(centering: Centering) -> int:
    codes = {"cell": 0, "vertex": 1, "vertex-interior": 2, "vertex-periodic": 3}
    return codes[centering.name]


def _centering_from_code(code: int) -> Centering:
    if code == 0:
        return CellCentered()
    if code == 1:
        return VertexCentered(include_boundary=True)
    if code == 2:
        return VertexCentered(include_boundary=False)
    if code == 3:
        return VertexCentered(periodic=True)
    raise ValueError(f"Unknown centering code {code}")


def _encode_boundary_rule(rule: BoundaryRule) -> bytes:
    code = _BC_TYPE_TO_CODE.get(type(rule))
    if code is None:
        raise ValueError(f"Cannot serialize boundary rule {rule!r}")
    wall_value = rule.value if isinstance(rule, Dirichlet) else 0.0
    return _BC.pack(code, wall_value)


def _decode_boundary_rule(buffer: bytes, pos: int) -> tuple[BoundaryRule, int]:
    code, wall_value = _BC.unpack_from(buffer, pos)
    pos += _BC.size
    rules: dict[int, BoundaryRule] = {
        0: Extrapolate(),
        1: Neumann(),
        2: Periodic(),
        3: Dirichlet(wall_value),
    }
    if code not in rules:
        raise ValueError(f"Unknown boundary rule code {code}")
    return rules[code], pos


def _encode_basis(basis: Basis1D) -> bytes:
    """One self-describing basis descriptor, see the module docstring."""
    out = struct.pack("<B", _centering_code(basis.centering))
    out += _encode_boundary_rule(basis.bc_left)
    out += _encode_boundary_rule(basis.bc_right)
    out += struct.pack("<B", _EVALUATION_TO_CODE[basis.scheme.evaluation])
    name = basis.scheme.name.encode("ascii")
    if len(name) > 15:
        raise ValueError(
            f"Scheme names are limited to 15 ascii characters: {basis.scheme.name!r}"
        )
    out += struct.pack("<16s", name)  # zero-terminated/-padded, c-style
    if len(basis.scheme.steps) > 255:
        raise ValueError("Too many lifting steps to serialize")
    out += struct.pack("<B", len(basis.scheme.steps))
    for step in basis.scheme.steps:
        if len(step.offsets) != len(step.weights):
            raise ValueError(f"Step has mismatched offsets and weights: {step!r}")
        if any(not -128 <= offset <= 127 for offset in step.offsets):
            raise ValueError(f"Stencil offsets must fit in int8, got {step.offsets}")
        out += _STEP.pack(_STEP_KIND_TO_CODE[step.kind], step.factor, len(step.offsets))
        for offset, weight in zip(step.offsets, step.weights):
            out += _TAP.pack(offset, weight)
    return out


def _decode_basis(buffer: bytes, pos: int) -> tuple[Basis1D, int]:
    (centering_code,) = struct.unpack_from("<B", buffer, pos)
    pos += 1
    centering = _centering_from_code(centering_code)
    bc_left, pos = _decode_boundary_rule(buffer, pos)
    bc_right, pos = _decode_boundary_rule(buffer, pos)
    (evaluation_code,) = struct.unpack_from("<B", buffer, pos)
    pos += 1
    if evaluation_code not in _CODE_TO_EVALUATION:
        raise ValueError(f"Unknown evaluation hint code {evaluation_code}")
    (raw_name,) = struct.unpack_from("<16s", buffer, pos)
    pos += 16
    name = raw_name.split(b"\0", 1)[0].decode("ascii")
    (num_steps,) = struct.unpack_from("<B", buffer, pos)
    pos += 1
    steps = []
    for _ in range(num_steps):
        kind_code, factor, num_taps = _STEP.unpack_from(buffer, pos)
        pos += _STEP.size
        if kind_code not in _CODE_TO_STEP_KIND:
            raise ValueError(f"Unknown lifting step kind code {kind_code}")
        offsets = []
        weights = []
        for _ in range(num_taps):
            offset, weight = _TAP.unpack_from(buffer, pos)
            pos += _TAP.size
            offsets.append(offset)
            weights.append(weight)
        steps.append(
            LiftingStep(
                _CODE_TO_STEP_KIND[kind_code],  # type: ignore[arg-type]
                tuple(offsets),
                tuple(weights),
                factor,
            )
        )
    scheme = LiftingScheme(
        name,
        tuple(steps),
        centering_kind="cell" if centering_code == 0 else "vertex",
        evaluation=_CODE_TO_EVALUATION[evaluation_code],  # type: ignore[arg-type]
    )
    # Basis1D.__post_init__ validates the centering/scheme/boundary pairing
    return Basis1D(centering, scheme, bc_left, bc_right), pos


def _encode_basis_block(bases: tuple[Basis1D, ...]) -> bytes:
    """The basis block including its uint32 length prefix."""
    uniform = all(basis == bases[0] for basis in bases)
    block = struct.pack("<B", 1 if uniform else 0)
    for basis in bases[:1] if uniform else bases:
        block += _encode_basis(basis)
    return struct.pack("<I", len(block)) + block


def _decode_basis_block(block: bytes, num_dims: int) -> tuple[Basis1D, ...]:
    (uniformity,) = struct.unpack_from("<B", block, 0)
    pos = 1
    if uniformity == 1:
        basis, pos = _decode_basis(block, pos)
        bases = per_dimension(basis, Basis1D, num_dims, "basis")
    elif uniformity == 0:
        collected = []
        for _ in range(num_dims):
            basis, pos = _decode_basis(block, pos)
            collected.append(basis)
        bases = tuple(collected)
    else:
        raise ValueError(f"Unknown basis uniformity flag {uniformity}")
    if pos != len(block):
        raise ValueError("Basis block has trailing bytes")
    return bases


_METADATA_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")
_TAG_STRING = 0
_TAG_FLOAT64 = 1
_TAG_INT64 = 2
_TAG_FLOAT64_ARRAY = 3
_TAG_INT64_ARRAY = 4
_TAG_BYTES = 5
_TAG_STRING_ARRAY = 6
_KNOWN_TAGS = frozenset(range(7))


def _encode_metadata_value(value: MetadataValue) -> tuple[int, bytes]:
    """Encode one metadata value as (tag, payload). Metadata is descriptive
    only, so the type surface is deliberately small and flat."""
    if isinstance(value, OpaqueValue):
        if value.tag in _KNOWN_TAGS:
            raise ValueError(
                f"OpaqueValue tag {value.tag} collides with a known tag; "
                "use the native type instead"
            )
        return value.tag, value.payload
    if isinstance(value, str):
        return _TAG_STRING, value.encode("utf-8")
    if isinstance(value, bytes):
        return _TAG_BYTES, value
    if isinstance(value, (bool, int, np.integer)):
        return _TAG_INT64, struct.pack("<q", int(value))
    if isinstance(value, (float, np.floating)):
        return _TAG_FLOAT64, struct.pack("<d", float(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, str) for item in value
        ):
            encoded_strings = [item.encode("utf-8") for item in value]
            if len(encoded_strings) > 65535 or any(
                len(item) > 65535 for item in encoded_strings
            ):
                raise ValueError("String array metadata entry is too large")
            payload = struct.pack("<H", len(encoded_strings))
            for item in encoded_strings:
                payload += struct.pack("<H", len(item)) + item
            return _TAG_STRING_ARRAY, payload
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError("Array metadata values must be one-dimensional")
        if array.dtype.kind == "f":
            return _TAG_FLOAT64_ARRAY, array.astype("<f8").tobytes()
        if array.dtype.kind in "iu":
            return _TAG_INT64_ARRAY, array.astype("<i8").tobytes()
        raise ValueError(f"Cannot serialize metadata array of dtype {array.dtype}")
    raise ValueError(f"Cannot serialize metadata value of type {type(value)!r}")


def _decode_metadata_value(tag: int, payload: bytes) -> MetadataValue:
    if tag == _TAG_STRING:
        return payload.decode("utf-8")  # strict: bad utf-8 raises
    if tag == _TAG_FLOAT64:
        if len(payload) != 8:
            raise ValueError("float64 metadata payload must be 8 bytes")
        return float(struct.unpack("<d", payload)[0])
    if tag == _TAG_INT64:
        if len(payload) != 8:
            raise ValueError("int64 metadata payload must be 8 bytes")
        return int(struct.unpack("<q", payload)[0])
    if tag == _TAG_FLOAT64_ARRAY:
        if len(payload) % 8 != 0:
            raise ValueError("float64 array metadata payload size mismatch")
        return tuple(float(x) for x in np.frombuffer(payload, dtype="<f8"))
    if tag == _TAG_INT64_ARRAY:
        if len(payload) % 8 != 0:
            raise ValueError("int64 array metadata payload size mismatch")
        return tuple(int(x) for x in np.frombuffer(payload, dtype="<i8"))
    if tag == _TAG_BYTES:
        return payload
    if tag == _TAG_STRING_ARRAY:
        (count,) = struct.unpack_from("<H", payload, 0)
        pos = 2
        strings = []
        for _ in range(count):
            if pos + 2 > len(payload):
                raise ValueError("String array metadata payload is truncated")
            (length,) = struct.unpack_from("<H", payload, pos)
            pos += 2
            if pos + length > len(payload):
                raise ValueError("String array metadata payload is truncated")
            strings.append(payload[pos : pos + length].decode("utf-8"))
            pos += length
        if pos != len(payload):
            raise ValueError("String array metadata payload has trailing bytes")
        return tuple(strings)
    # unknown tag: preserve verbatim, never interpret
    return OpaqueValue(tag, payload)


def _validate_metadata_key(key: str) -> bytes:
    if not 1 <= len(key) <= 63 or not set(key) <= _METADATA_KEY_CHARS:
        raise ValueError(
            f"Metadata keys must be 1-63 characters of [a-z0-9_.-], got {key!r}"
        )
    return key.encode("ascii")


def _encode_metadata_block(metadata: "dict[str, MetadataValue]") -> bytes:
    """The metadata block including its uint32 length prefix."""
    if len(metadata) > 65535:
        raise ValueError("Too many metadata entries")
    block = struct.pack("<H", len(metadata))
    for key, value in metadata.items():
        encoded_key = _validate_metadata_key(key)
        tag, payload = _encode_metadata_value(value)
        block += struct.pack("<B", len(encoded_key)) + encoded_key
        block += struct.pack("<BI", tag, len(payload)) + payload
    return struct.pack("<I", len(block)) + block


def _decode_metadata_block(block: bytes) -> "dict[str, MetadataValue]":
    # everything is parsed from the already-read block buffer; every length
    # is validated against it before slicing, so corrupted lengths cannot
    # over-read or drive allocations
    (num_entries,) = struct.unpack_from("<H", block, 0)
    pos = 2
    metadata: dict[str, MetadataValue] = {}
    for _ in range(num_entries):
        if pos + 1 > len(block):
            raise ValueError("Metadata block is truncated")
        (key_length,) = struct.unpack_from("<B", block, pos)
        pos += 1
        if not 1 <= key_length <= 63 or pos + key_length > len(block):
            raise ValueError("Invalid metadata key length")
        key = block[pos : pos + key_length].decode("ascii")
        if not set(key) <= _METADATA_KEY_CHARS:
            raise ValueError(f"Invalid characters in metadata key {key!r}")
        pos += key_length
        if pos + 5 > len(block):
            raise ValueError("Metadata block is truncated")
        tag, payload_length = struct.unpack_from("<BI", block, pos)
        pos += 5
        if pos + payload_length > len(block):
            raise ValueError("Metadata payload exceeds its block")
        payload = block[pos : pos + payload_length]
        pos += payload_length
        if key in metadata:
            raise ValueError(f"Duplicate metadata key {key!r}")
        metadata[key] = _decode_metadata_value(tag, payload)
    if pos != len(block):
        raise ValueError("Metadata block has trailing bytes")
    return metadata


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


def open_file(path: "str | Path", mode: str = "rb") -> BinaryIO:
    """Small wrapper for file access used by the prototype API."""
    return cast(BinaryIO, Path(path).open(mode))


def _open_stream(target: "str | Path | BinaryIO", mode: str) -> tuple[BinaryIO, bool]:
    if isinstance(target, (str, Path)):
        return open_file(target, mode), True
    return target, False


def _stored_entries(data: Tensor) -> "tuple[np.ndarray, np.ndarray]":
    """The nonzero (sorted keys, values) of the tensor. Explicit zeros of
    sparse-style tensors are dropped, so a tensor whose values were partly
    reset to zero re-encodes as if those entries had never been stored."""
    if data.is_sparse:
        keys, values = data.linear_indices, data.linear_values
        nonzero = values != 0
        if not bool(nonzero.all()):
            keys, values = keys[nonzero], values[nonzero]
        return keys, values
    keys = np.flatnonzero(data.linear_values)
    return keys, data.linear_values[keys]


def _bridged_runs(
    keys: "np.ndarray", index_itemsize: int, value_itemsize: int
) -> "tuple[np.ndarray, np.ndarray]":
    """Runs of the given nonzero positions, with short zero gaps bridged:
    storing a gap's zeros explicitly costs gap * value_itemsize bytes,
    splitting costs one extra pair of run bounds -- so gaps of up to
    2 * index_itemsize / value_itemsize zeros are cheaper (or equal, with
    fewer runs) inside a run. Once values are stored bit-packed, their
    per-entry cost shrinks to precision_bits / 8, admitting even longer
    bridges in this same inequality."""
    firsts, lasts = runs_from_sorted_keys(keys)
    if firsts.size <= 1:
        return firsts, lasts
    gaps = firsts[1:] - lasts[:-1] - 1
    split_after = np.flatnonzero(gaps * value_itemsize > 2 * index_itemsize)
    merged_firsts = firsts[np.concatenate(([0], split_after + 1))]
    merged_lasts = lasts[np.concatenate((split_after, [lasts.size - 1]))]
    return merged_firsts, merged_lasts


def _cheapest_kind(
    size: int, itemsize: int, num_nonzero: int, num_runs: int, num_covered: int
) -> TensorKind:
    """The on-disk kind with the smallest data blob; ties go to the
    simpler kind (FULL, then POINTWISE, then INTERVALS)."""
    index_itemsize = _index_dtype(size).itemsize
    kind = TensorKind.FULL
    cheapest = size * itemsize
    sparse_bytes = num_nonzero * (index_itemsize + itemsize)
    if sparse_bytes < cheapest:
        kind, cheapest = TensorKind.POINTWISE, sparse_bytes
    interval_bytes = 2 * num_runs * index_itemsize + num_covered * itemsize
    if interval_bytes < cheapest:
        kind = TensorKind.INTERVALS
    return kind


def _encode_record(subspace: Subspace, num_dims: int) -> bytes:
    if subspace.compression != 0:
        raise ValueError(
            f"Reserved compression byte must be 0, got {subspace.compression}"
        )
    if subspace.quantization_scale != 1.0 or subspace.quantization_zero_point != 0:
        raise ValueError(
            "The quantization fields are reserved: only identity "
            "(scale=1, zero_point=0) can be serialized"
        )
    if subspace.num_components != 1:
        raise ValueError(
            "num_components is reserved: only 1 can be serialized, "
            f"got {subspace.num_components}"
        )
    if subspace.component_layout != 0:
        raise ValueError(
            "component_layout is reserved: only 0 (planes) can be "
            f"serialized, got {subspace.component_layout}"
        )
    if any(not 0 <= extent <= 0xFFFFFFFF for extent in subspace.extents):
        raise ValueError(
            f"Extents must fit in uint32 each, got {subspace.extents} "
            "(this caps levels at 31 per dimension)"
        )
    extents = struct.pack(f"<{num_dims}I", *subspace.extents)
    data = subspace.data
    if data is None:
        kind = TensorKind.EMPTY
        value_dtype = np.dtype(np.float64)  # placeholder, nothing is stored
        num_stored = 0
        blob = b""
    else:
        value_dtype = data.dtype
        little_endian = value_dtype.newbyteorder("<")
        # store whichever representation yields the smallest data blob,
        # independently of the in-memory tensor kind (implicit and explicit
        # zeros read back identically)
        keys, values = _stored_entries(data)
        index_dtype = _index_dtype(data.size)
        run_firsts, run_lasts = _bridged_runs(
            keys, index_dtype.itemsize, value_dtype.itemsize
        )
        num_covered = int((run_lasts - run_firsts + 1).sum())
        kind = _cheapest_kind(
            data.size, value_dtype.itemsize, len(values), len(run_firsts), num_covered
        )
        if kind == TensorKind.FULL:
            if data.is_sparse:
                flat = np.zeros(data.size, dtype=value_dtype)
                flat[keys] = values
            else:
                flat = data.linear_values
            num_stored = data.size
            blob = flat.astype(little_endian, copy=False).tobytes()
        elif kind == TensorKind.POINTWISE:
            num_stored = len(values)
            blob = (
                keys.astype(index_dtype).tobytes()
                + values.astype(little_endian, copy=False).tobytes()
            )
        else:  # TensorKind.INTERVALS
            # scatter the nonzeros into the covered buffer
            covered_values = np.zeros(num_covered, dtype=value_dtype)
            offsets = np.concatenate(([0], np.cumsum(run_lasts - run_firsts + 1)[:-1]))
            run = np.searchsorted(run_firsts, keys, side="right") - 1
            covered_values[offsets[run] + keys - run_firsts[run]] = values
            num_stored = num_covered
            blob = (
                run_firsts.astype(index_dtype).tobytes()
                + run_lasts.astype(index_dtype).tobytes()
                + covered_values.astype(little_endian, copy=False).tobytes()
            )
    if not 1 <= subspace.precision_bits <= 8 * value_dtype.itemsize:
        raise ValueError(
            f"precision_bits must be in [1, {8 * value_dtype.itemsize}] for "
            f"dtype {value_dtype}, got {subspace.precision_bits} (the "
            "container dtype must be wide enough for the stored format)"
        )
    number_format = subspace.number_format
    assert number_format is not None  # normalized in Subspace.__post_init__
    fixed = _RECORD.pack(
        _ORDER_TO_CODE[subspace.order],
        int(kind),
        value_dtype.kind.encode("ascii"),
        value_dtype.itemsize,
        subspace.num_components,
        subspace.component_layout,
        subspace.precision_bits,
        number_format.to_byte(),
        subspace.padding_bits,
        subspace.compression,
        subspace.quantization_scale,
        subspace.quantization_zero_point,
        num_stored,
        len(blob),
    )
    # the blob checksum follows the blob
    return extents + fixed + blob + _RECORD_CRC.pack(zlib.crc32(blob))


def _decode_record(stream: BinaryIO, num_dims: int) -> Subspace:
    extents = struct.unpack(f"<{num_dims}I", _read_exactly(stream, 4 * num_dims))
    (
        order_code,
        kind_code,
        dtype_kind,
        itemsize,
        num_components,
        component_layout,
        precision_bits,
        number_format_byte,
        padding_bits,
        compression,
        quantization_scale,
        quantization_zero_point,
        num_stored,
        num_blob_bytes,
    ) = _RECORD.unpack(_read_exactly(stream, _RECORD.size))

    # validate every field the blob size derives from BEFORE reading the
    # blob, so a corrupted size field cannot drive a huge allocation, and
    # so all rejections surface as ValueError (the documented contract)
    order = _CODE_TO_ORDER.get(order_code)
    if order is None:
        raise ValueError(f"Unknown linearization order code {order_code}")
    kind = TensorKind(kind_code)  # raises ValueError for unknown codes
    if compression != 0:
        raise ValueError(f"Reserved compression byte must be 0, got {compression}")
    if quantization_scale != 1.0 or quantization_zero_point != 0:
        raise ValueError(
            "Reserved quantization fields must be identity "
            f"(scale=1, zero_point=0), got scale={quantization_scale}, "
            f"zero_point={quantization_zero_point}"
        )
    if num_components != 1:
        raise ValueError(
            f"Reserved num_components byte must be 1, got {num_components}"
        )
    if component_layout != 0:
        raise ValueError(
            f"Reserved component_layout byte must be 0, got {component_layout}"
        )
    shape = tuple(int(e) for e in extents)
    total = math.prod(shape) if shape else 1  # Python ints: no int64 overflow
    try:
        value_dtype = np.dtype(f"{dtype_kind.decode('ascii')}{itemsize}")
    except (TypeError, UnicodeDecodeError) as error:
        raise ValueError(
            f"Invalid value dtype {dtype_kind!r} with itemsize {itemsize}"
        ) from error
    if not 1 <= precision_bits <= 8 * itemsize:
        raise ValueError(
            f"precision_bits must be in [1, {8 * itemsize}] for "
            f"dtype {value_dtype}, got {precision_bits}"
        )
    # raises ValueError for structurally invalid flavor bytes
    number_format = NumberFormat.from_byte(number_format_byte)
    if (0 if number_format.unsigned else 1) + number_format.exponent_bits > (
        precision_bits
    ):
        raise ValueError(
            f"Number format {number_format} does not fit in "
            f"{precision_bits} precision bits"
        )

    if kind == TensorKind.EMPTY:
        if num_stored != 0:
            raise ValueError("Empty subspace record must store zero entries")
        expected_blob_bytes = 0
    elif kind == TensorKind.FULL:
        if num_stored != total:
            raise ValueError("Full subspace record must store every entry")
        expected_blob_bytes = total * itemsize
    elif kind == TensorKind.POINTWISE:
        if num_stored > total:
            raise ValueError("Sparse subspace stores more entries than it has")
        index_dtype = _index_dtype(total)
        expected_blob_bytes = num_stored * (index_dtype.itemsize + itemsize)
    else:  # TensorKind.INTERVALS
        if num_stored > total:
            raise ValueError("Interval subspace stores more entries than it has")
        index_dtype = _index_dtype(total)
        # the run count is derived from the blob size; every constraint is
        # checked before the blob is read
        bounds_bytes = num_blob_bytes - num_stored * itemsize
        if bounds_bytes < 0 or bounds_bytes % (2 * index_dtype.itemsize) != 0:
            raise ValueError("Subspace data block has inconsistent size")
        num_runs = bounds_bytes // (2 * index_dtype.itemsize)
        if num_runs > num_stored or (num_runs == 0) != (num_stored == 0):
            raise ValueError("Interval subspace has an inconsistent run count")
        expected_blob_bytes = num_blob_bytes
    if num_blob_bytes != expected_blob_bytes:
        raise ValueError("Subspace data block has inconsistent size")

    blob = _read_exactly(stream, expected_blob_bytes)
    (checksum,) = _RECORD_CRC.unpack(_read_exactly(stream, _RECORD_CRC.size))
    if zlib.crc32(blob) != checksum:
        raise ValueError("Subspace data block failed its checksum")

    data: Tensor | None
    if kind == TensorKind.EMPTY:
        data = None
    elif kind == TensorKind.FULL:
        # astype() also yields a writable copy of the read-only buffer view
        flat = np.frombuffer(blob, dtype=value_dtype.newbyteorder("<")).astype(
            value_dtype
        )
        data = DenseTensor(flat, shape, order=order)
    elif kind == TensorKind.POINTWISE:
        split = num_stored * index_dtype.itemsize
        keys = np.frombuffer(blob[:split], dtype=index_dtype).astype(np.int64)
        values = np.frombuffer(
            blob[split:], dtype=value_dtype.newbyteorder("<")
        ).astype(value_dtype)
        # SparseTensor validates key range and uniqueness
        data = SparseTensor.from_linear(keys, values, shape, order=order)
    else:  # TensorKind.INTERVALS
        split = num_runs * index_dtype.itemsize
        run_firsts = np.frombuffer(blob[:split], dtype=index_dtype).astype(np.int64)
        run_lasts = np.frombuffer(blob[split : 2 * split], dtype=index_dtype).astype(
            np.int64
        )
        values = np.frombuffer(
            blob[2 * split :], dtype=value_dtype.newbyteorder("<")
        ).astype(value_dtype)
        # IntervalTensor validates bounds, disjointness, and maximality
        data = IntervalTensor(run_firsts, run_lasts, values, shape, order=order)

    return Subspace(
        extents=shape,
        precision_bits=precision_bits,
        data=data,
        number_format=number_format,
        num_components=num_components,
        component_layout=component_layout,
        quantization_scale=quantization_scale,
        quantization_zero_point=quantization_zero_point,
        padding_bits=padding_bits,
        compression=compression,
    )


def write(
    tensors: SparseGridHierarchicalTensors, target: "str | Path | BinaryIO"
) -> None:
    """Write the hierarchy to a path or binary stream in the v0.5 layout."""
    num_dims = tensors.dimensions
    if not 1 <= num_dims <= 65535:
        raise ValueError(f"Number of dimensions must fit in uint16, got {num_dims}")
    for level in (*tensors.subspaces.keys(), tensors.max_level, tensors.min_level):
        if any(not 0 <= single_level <= 255 for single_level in level):
            raise ValueError(f"Levels must fit in one byte each, got {level}")

    records = [
        _encode_record(subspace, num_dims) for subspace in tensors.subspaces.values()
    ]
    basis_block = _encode_basis_block(tensors.bases)  # includes its length prefix
    metadata_block = _encode_metadata_block(tensors.metadata)  # ditto
    table_entry = struct.Struct(f"<{num_dims}BQ")
    header_size = (
        _HEADER.size
        + 2 * num_dims  # max level + min level
        + len(basis_block)
        + len(metadata_block)
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
        + basis_block
        + metadata_block
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
        # table offsets are relative to the payload start
        payload_start = stream.tell()
        fixed_header = _read_exactly(stream, _HEADER.size)
        magic, major, minor, num_dims, num_subspaces = _HEADER.unpack(fixed_header)
        if magic != FormatMagic:
            raise ValueError("Not a spght file (magic string mismatch)")
        if (major, minor) != FormatVersion:
            raise ValueError(
                f"Unsupported format version {major}.{minor}, "
                f"expected {FormatVersion[0]}.{FormatVersion[1]}"
            )

        levels_bytes = _read_exactly(stream, 2 * num_dims)
        basis_length_bytes = _read_exactly(stream, 4)
        (basis_length,) = struct.unpack("<I", basis_length_bytes)
        basis_block = _read_exactly(stream, basis_length)
        metadata_length_bytes = _read_exactly(stream, 4)
        (metadata_length,) = struct.unpack("<I", metadata_length_bytes)
        metadata_block = _read_exactly(stream, metadata_length)
        table_entry = struct.Struct(f"<{num_dims}BQ")
        table_bytes = _read_exactly(stream, num_subspaces * table_entry.size)
        (header_crc,) = _HEADER_CRC.unpack(_read_exactly(stream, _HEADER_CRC.size))
        if (
            zlib.crc32(
                fixed_header
                + levels_bytes
                + basis_length_bytes
                + basis_block
                + metadata_length_bytes
                + metadata_block
                + table_bytes
            )
            != header_crc
        ):
            raise ValueError("File header failed its checksum")

        max_level = struct.unpack(f"<{num_dims}B", levels_bytes[:num_dims])
        min_level = struct.unpack(f"<{num_dims}B", levels_bytes[num_dims:])
        bases = _decode_basis_block(basis_block, num_dims)
        metadata = _decode_metadata_block(metadata_block)
        table: list[tuple[tuple[int, ...], int]] = []
        for i in range(num_subspaces):
            entry = table_entry.unpack_from(table_bytes, i * table_entry.size)
            table.append((entry[:-1], entry[-1]))

        subspaces: dict[tuple[int, ...], Subspace] = {}
        for level, offset in table:
            stream.seek(payload_start + offset)
            subspaces[level] = _decode_record(stream, num_dims)
        if len(subspaces) != num_subspaces:
            raise ValueError("Duplicate subspace levels in the table")
    finally:
        if should_close:
            stream.close()

    return SparseGridHierarchicalTensors(
        dimensions=num_dims,
        max_level=tuple(max_level),
        min_level=tuple(min_level),
        bases=bases,
        subspaces=subspaces,
        metadata=metadata,
    )
