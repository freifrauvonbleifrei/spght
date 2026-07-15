<!--
SPDX-FileCopyrightText: 2026 Theresa Pollinger

SPDX-License-Identifier: CC-BY-4.0
-->

# spght: Sparse Grid Hierarchical Tensors

[![Python Lint and Test](https://github.com/freifrauvonbleifrei/spght/actions/workflows/python-lint-and-test.yml/badge.svg)](https://github.com/freifrauvonbleifrei/spght/actions/workflows/python-lint-and-test.yml)
[![WDAS Cloud Compression](https://github.com/freifrauvonbleifrei/spght/actions/workflows/python-example.yml/badge.svg)](https://github.com/freifrauvonbleifrei/spght/actions/workflows/python-example.yml)
[![Coverage](./coverage.svg)](https://github.com/freifrauvonbleifrei/spght/actions/workflows/python-coverage.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSES/Apache-2.0.txt)

pronounced "spaghetti".

spght defines a memory and storage representation of sparse grids /
hierarchical wavelet coefficients. A function on a structured grid is
`hierarchize`d into hierarchical subspaces (one coefficient block per level
vector), optionally `compress`ed by dropping small coefficients, evaluated
with `interpolate`, and written to / read from the binary file format
documented below.

## Installation
Install straight from GitHub:

```shell
pip install git+https://github.com/freifrauvonbleifrei/spght.git
```

or, from a local checkout (editable, for development):

```shell
git clone https://github.com/freifrauvonbleifrei/spght.git
cd spght
pip install -e .
```

spght depends only on `numpy` and `bitarray`. The optional `[pywt]` extra
(`pip install -e ".[pywt]"`) adds the PyWavelets bridge in
`spght.pywt_compat`, which shows how the half-normalized Haar transform
maps onto the classical filter-bank machinery; without it, everything
except that bridge works identically.

Run the tests with `pip install pytest && pytest test/`.

## Usage

The whole pipeline is reachable from the top-level namespace:

```python
import numpy as np
import spght

# a function sampled on a structured grid; the extents must match the
# basis' centering: 2**l cells (default Haar / periodic vertex grids),
# 2**l + 1 vertices (e.g. spght.hat_basis()), or 2**l - 1 interior vertices
nodal_values = np.random.default_rng(0).random((64, 64, 64))

# decompose into hierarchical subspaces, one coefficient block per level vector
tensors = spght.hierarchize(nodal_values)

# drop coefficients with |coefficient| <= epsilon
compressed = spght.compress(tensors, epsilon=0.01)

# optional: re-linearize the coefficient buffers, e.g. onto Z-order curves
compressed.relinearize("ZC")

# write to / read from the .spght binary format (path or binary stream);
# equivalently: compressed.write(...) and spght.SparseGridHierarchicalTensors.read(...)
spght.write(compressed, "function.spght")
loaded = spght.read("function.spght")

# point-evaluate on the unit hypercube [0, 1]^d
value = spght.interpolate(np.array([0.3, 0.6, 0.5]), loaded)
```

For a complete worked example — compressing the WDAS cloud dataset and
evaluating the reconstruction error — see [`example/README.md`](example/README.md).

## File format, version 0.5

One spght file stores one `SparseGridHierarchicalTensors` container: a set of
subspaces, each identified by its level vector `l = (l_1, ..., l_d)` and
holding a linear buffer of coefficient values.

General properties:

- All multi-byte values are **little endian**. There is no implicit padding
  or alignment; every field follows the previous one directly.
- Subspaces appear (in the table and as records) in the **canonical order**:
  ascending level sum, ties broken lexicographically with dimension 0 most
  significant. A file prefix therefore contains a complete coarse
  approximation that is progressively refined by further records.
- Every subspace's values are stored as a **1-D buffer in the subspace's
  linearization order** (see *Linearization orders*); the n-dimensional view
  is reconstructed from extents + order on demand.
- The format is a **draft**: it may change without a version bump until 1.0.

### File header

| offset | size (bytes) | type | field |
|---|---|---|---|
| 0 | 33 | bytes | magic string: ASCII `"sparse grid hierarchical tensors"` followed by one NUL byte |
| 33 | 1 | uint8 | format version, major (currently 0) |
| 34 | 1 | uint8 | format version, minor (currently 5) |
| 35 | 2 | uint16 | number of dimensions `d` (1 to 65535) |
| 37 | 8 | uint64 | number of subspaces `n` |
| 45 | `d` | uint8 each | maximum level per dimension |
| 45 + `d` | `d` | uint8 each | minimum level per dimension (see *Scaling vs. detail subspaces*) |
| 45 + 2`d` | 4 | uint32 | byte length `b` of the basis block that follows (readers may skip it wholesale) |
| 45 + 2`d` + 4 | `b` | basis block | uniformity flag + basis descriptor(s), see *Basis descriptors* |
| 45 + 2`d` + 4 + `b` | 4 | uint32 | byte length `m` of the metadata block that follows (skippable wholesale) |
| ... | `m` | metadata block | uint16 entry count + key-value entries, see *Metadata* |
| ... | `n * (d + 8)` | table entries | subspace table (see below) |
| ... | 4 | uint32 | CRC-32 (zlib) checksum of all preceding header bytes, verified on read |

Each **subspace table** entry is:

| size (bytes) | type | field |
|---|---|---|
| `d` | uint8 each | level vector of the subspace |
| 8 | uint64 | byte offset of the subspace record, relative to the payload start (the first magic byte) — identical to an absolute offset when the payload starts at byte 0 |

The table makes every record independently addressable, so a reader can
select subspaces by level (e.g. to interpolate only at relevant scales)
without scanning the file. Records are laid out contiguously after the
header, in table order. Payload-relative offsets make embedding safe: a
`.spght` payload stored at any position inside a larger file reads
correctly once the reader seeks to its first magic byte.

### Subspace record

| size (bytes) | type | field |
|---|---|---|
| `4 * d` | uint32 each | extents (logical shape of the subspace) |
| 1 | uint8 | linearization order code: `C` = 0, `F` = 1, `ZC` = 2, `ZF` = 3 |
| 1 | uint8 | tensor kind: `EMPTY` = 0, `FULL` = 1, `POINTWISE` = 2 |
| 1 | char | value dtype: numpy kind character (`f` float, `i` signed int, `u` unsigned int, ...) |
| 1 | uint8 | value dtype: item size in bytes (together e.g. `f8` = float64, `i1` = int8); the dtype describes **one scalar component** and is authoritative for decoding — it is never multiplexed with a component count |
| 1 | uint8 | `num_components` (must be 1; reserved, see below) |
| 1 | uint8 | `component_layout` (must be 0; reserved, see below) |
| 1 | uint8 | precision bits: total bits of the stored number format, per scalar component (see *Value dtype and number format*) |
| 1 | uint8 | number format: bits 0–4 exponent width, bit 5 unsigned flag, bits 6–7 special-value convention (see *Value dtype and number format*) |
| 2 | uint16 | padding bits (trailing slack bits; meaningful only under a future bit-packing transform, carried verbatim until then) |
| 1 | uint8 | compression (0 = none; reserved) |
| 4 | float32 | `quantization_scale` (reserved, see below) |
| 4 | int32 | `quantization_zero_point` (reserved) |
| 8 | uint64 | number of stored entries |
| 8 | uint64 | number of bytes in the data blob |
| (blob) | bytes | data blob, see below |
| 4 | uint32 | CRC-32 (zlib) checksum of the data blob, verified on read — placed *after* the blob so a writer can stream blob chunks while accumulating the checksum |

### Data blob

The blob content depends on the tensor kind:

- **`EMPTY` (0)**: no blob (zero bytes); every entry of the subspace is
  implicitly zero. This is deliberately the all-zero-bytes default: a
  zero-initialized record reads as "no data".
- **`FULL` (1)**: the complete linear value buffer, `prod(extents)` entries
  of the value dtype, in the subspace's linearization order.
- **`POINTWISE` (2)**: the individual sorted linear indices of the stored entries, followed
  by the matching value buffer. Indices are stored in the **smallest unsigned
  integer type that can hold `prod(extents) - 1`** (uint8 up to 256 entries,
  uint16 up to 2^16, uint32 up to 2^32, else uint64); this type is derived
  from the extents and not stored explicitly. Entries not listed are
  implicitly zero.

Future kinds (e.g. interval/run-based sparsity) get new tensor-kind values.

### Value dtype and number format

The value dtype is the **storage container**: it alone determines how the
blob is sliced into scalars and decoded, and it is restricted to standard
(numpy-native) formats, so every reader decodes with native machinery.
The two bytes that follow describe the **number format the values live
on** — parametrically, so that new low-precision formats need no change
to this specification:

- **precision bits** (1 to `8 * itemsize`): the total bit width of the
  format.
- **number format** (the flavor byte): bits 0–4 hold the exponent width
  `e` (0 = integer, no exponent), bit 5 the unsigned flag (no sign bit),
  bits 6–7 the special-value convention. The mantissa width is *derived*
  (`precision − sign − e`), and the exponent bias is *implied by the
  convention*:

  | convention | meaning | bias |
  |---|---|---|
  | `IEEE` (0) | infinities and NaNs at the all-ones exponent | `2^(e−1) − 1` |
  | `FN` (1) | finite only: no infinities, a single NaN pattern | `2^(e−1) − 1` |
  | `FNUZ` (2) | finite, NaN at the negative-zero pattern | `2^(e−1)` |
  | `OTHER` (3) | not parametric (stored canonically: `e = 0`, signed) | — |

Any sign/exponent/mantissa format is expressible without being named
here: float64 is (64, e11, IEEE), bfloat16 (16, e8, IEEE), tf32 (19, e8,
IEEE), float8-E5M2 (8, e5, IEEE), float8-E4M3FN (8, e4, FN), the
FNUZ fp8 variants, fp4 E2M1 (4, e2, IEEE), the E8M0 scale format
(8, e8, unsigned, FN), and int/uint of any width (`e = 0`).

Crucially, the values always travel **widened losslessly into the
container dtype** (every fp8/fp4 value is exactly representable in
float16): the flavor bytes never change how bytes are decoded, they
declare which grid the decoded values lie on. A reader that ignores
them — or meets `OTHER`, the escape hatch for non-parametric formats
such as posits, whose provenance a metadata entry may record — still
reads exact values; only transcoding, re-quantization, and rate
accounting need the flavor. Storing narrow formats *tightly* (paying 8
bits on disk, not 16) is a future transform behind the reserved
`compression` byte — packing the raw `precision`-wide bit patterns back
to back, `padding_bits` closing the final byte — again parametric, one
transform for every flavor. Block-scaled formats (e.g. MXFP4's shared
scale per 32 values) are *not* a number format: they are a future
granularity extension of the quantization fields, composing with the
flavor byte.

### Basis descriptors

The basis block records the wavelet of the hierarchical transform. It starts
with one uint8 **uniformity flag**: 1 means a single basis descriptor follows
and applies to every dimension; 0 means `d` descriptors follow, one per
dimension. Each descriptor encodes the complete lifting program, so any
reader can reconstruct without a registry of named wavelets (the scheme name
travels as a label only, never as semantics):

| size (bytes) | type | field |
|---|---|---|
| 1 | uint8 | centering: cell = 0, vertex = 1, vertex-interior = 2, vertex-periodic = 3 |
| 1 + 8 | uint8 + float64 | left boundary rule: Extrapolate = 0, Neumann = 1, Periodic = 2, Dirichlet = 3; the float64 is the Dirichlet wall value (0.0 otherwise) |
| 1 + 8 | uint8 + float64 | right boundary rule, as above |
| 1 | uint8 | evaluation hint: none = 0, midpoint = 1, nodal_linear = 2 |
| 16 | bytes | scheme name: zero-terminated ASCII, C-style (at most 15 characters, NUL-padded; label only) |
| 1 | uint8 | number of lifting steps, then per step: |
| 1 | uint8 | step kind: predict = 0, update = 1, scale_detail = 2 |
| 8 | float64 | step factor |
| 1 | uint8 | number of taps, then per tap: |
| 1 + 8 | int8 + float64 | stencil offset and weight |

Descriptors are variable-sized (steps and taps differ between schemes) but
self-delimiting; the block's uint32 length prefix lets readers that only
need the subspace table skip it without parsing. The centering also
determines the dof counts per level (e.g. `2^l` cells vs `2^l + 1`
vertices), which is why extents in this format are always stored
explicitly.

### Metadata

The metadata block carries descriptive key-value entries: what the field
is, where it lives, who wrote it. Its design rule is security-relevant:
**metadata never influences how the rest of the file is parsed** — every
structural fact lives in the typed header fields, so a reader that skips
the block (via its length prefix) loses labels, never correctness, and a
corrupted or malicious block can at worst mislabel data.

Each entry is `key length (uint8, 1-63)` + ASCII key (charset
`[a-z0-9_.-]`, unique) + `value tag (uint8)` + `payload length (uint32)` +
payload. Value tags: UTF-8 string = 0, float64 = 1, int64 = 2, float64
array = 3, int64 array = 4, bytes = 5, string array = 6 (uint16 count,
then per string a uint16 byte length + UTF-8 bytes). Every length is
validated against the enclosing block before it is read.

Entries with unknown tags are preserved verbatim across read-write cycles
(surfaced as `spght.OpaqueValue`) and never interpreted — new value kinds
are therefore additive. Un-prefixed keys are reserved for this
specification; suggested (all optional): `field_name` (string),
`dimension_names` (string array), `domain_min`/`domain_max` (float64
array), `time` (float64), `created_by` (string). Applications should
prefix their own keys, e.g. `x-`. spght itself never acts on metadata
values: no paths are resolved, nothing is fetched or executed.

### Scaling vs. detail subspaces

The minimum level (header) is where the wavelet cascade stops in each
dimension. Subspace level labels are absolute: along dimension `d` they run
from `min_level[d]` to `max_level[d]`, and

- a subspace with `level[d] == min_level[d]` holds **scaling**
  (approximation) coefficients along `d`, with extent `2^min_level[d]`;
- a subspace with `level[d] > min_level[d]` holds **detail** (wavelet)
  coefficients along `d`, with extent `2^(level[d] - 1)`.

Scaling-ness is *defined* by this equality and not stored separately: for
this format, a dimension being at its minimum level and holding scaling
coefficients are the same thing. `min_level` of all zeros reproduces the
classic full decomposition.

### Normalization / quantization (reserved)

Each subspace record carries two fields reserved for future per-subspace
quantization support, with the intended semantics
`logical = quantization_scale * (stored - quantization_zero_point)`: a
float32 scale (default 1.0) and an int32 zero-point in units of the scale
(default 0).. Both writer and reader enforce the identity
values (scale = 1, zero_point = 0): a file with anything else is currently rejected with an error. 

### Vector-valued coefficients (reserved)

Each subspace record carries two bytes reserved for vector-valued
coefficients (e.g. multiwavelet / modal coefficients of a single field):
`num_components` values are stored per spatial point, arranged as
declared by `component_layout`:

- **planes (0)**, structure-of-arrays: the blob holds one complete
  spatial linearization per component, back to back. Favors
  per-component access and compressibility.
- **interleaved (1)**, array-of-structures: the components of each point
  are stored together, points following the linearization order. Favors
  streaming reconstruction (a reader emits complete vectors as bytes
  arrive) and in-place modification of a point's vector.

In both layouts the component axis participates in neither the
hierarchical transform nor the linearization order, and `POINTWISE`
indices stay spatial (one index selects a whole vector). The value dtype
and `precision_bits` always describe a single scalar component. 
Both writer and reader currently enforce
`num_components == 1` and `component_layout == 0`; a file with anything
else is rejected with an error. A `.spght` file holds one (possibly
vector-valued) field — several distinct physical fields are stored as
separate files.

### Linearization orders

The order code describes how the n-dimensional subspace is flattened into
the linear buffer (and what the `POINTWISE` indices refer to):

- **`C` (0)**: row-major, last dimension varies fastest.
- **`F` (1)**: column-major, first dimension varies fastest.
- **`ZC` (2)** / **`ZF` (3)**: Z-order (Morton) curves; dimensions are visited
  round-robin starting with dimension 0 (`ZC`) or the last dimension (`ZF`),
  where a dimension drops out of the rotation once its extent is exhausted.
  For power-of-two extents this is classic bit interleaving; for arbitrary
  extents the curve bisects each dimension's remaining extent into
  ceil/floor halves, staying bijective without padding.

### Streaming writes

The header-first layout is deliberate: a file *prefix* is a complete
coarse approximation (see the canonical order above), which a
footer/trailer layout would sacrifice. Single-pass writers are still
possible on seekable outputs: the header's *size* is known before any
record is written (levels, bases, and metadata are fixed up front), so a
writer can reserve the header region, stream the records — each record's
sizes are known when it starts, and the blob checksum follows the blob —
and finally seek back to patch the subspace table and header checksum.
For parallel (MPI-IO) writes, uncompressed FULL record sizes are
deterministic from extents and dtype, so all ranks can compute their
offsets without communication; one rank finalizes the header.

### Integrity and limits

- A reader must verify the magic string, reject unknown major/minor
  versions, verify the header CRC-32 before trusting the subspace table, and
  verify each record's CRC-32 before trusting its blob.
- The number of dimensions is a uint16: 1 to 65535. Levels are stored as
  single bytes. Extents are uint32 per dimension -- which effectively
  caps levels to 32 per dimension -- and byte counts are uint64.
- The precision and number format bytes never change how blob bytes are
  decoded (the dtype does); writer and reader both reject precision
  outside `[1, 8 * itemsize]`, structurally invalid flavor bytes, and
  formats whose sign + exponent bits exceed the precision.
  `compression` is reserved and must currently be 0.

## License

The spght code is licensed under [Apache-2.0](LICENSES/Apache-2.0.txt);
documentation and configuration files are [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)
or [CC0-1.0](LICENSES/CC0-1.0.txt) as marked. The repository follows the
[REUSE](https://reuse.software/) license specification.
