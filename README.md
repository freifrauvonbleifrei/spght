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

# write to / read from the .spght binary format (path or binary stream);
# equivalently: compressed.write(...) and spght.SparseGridHierarchicalTensors.read(...)
spght.write(compressed, "function.spght")
loaded = spght.read("function.spght")

# point-evaluate on the unit hypercube [0, 1]^d
value = spght.interpolate(np.array([0.3, 0.6, 0.5]), loaded)
```

For a complete worked example — compressing the WDAS cloud dataset and
evaluating the reconstruction error — see [`example/README.md`](example/README.md).

## File format, version 0.3

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
| 34 | 1 | uint8 | format version, minor (currently 3) |
| 35 | 2 | uint16 | number of dimensions `d` (1 to 65535) |
| 37 | 8 | uint64 | number of subspaces `n` |
| 45 | `d` | uint8 each | maximum level per dimension |
| 45 + `d` | `d` | uint8 each | minimum level per dimension (see *Scaling vs. detail subspaces*) |
| 45 + 2`d` | 4 | uint32 | byte length `b` of the basis block that follows (readers may skip it wholesale) |
| 45 + 2`d` + 4 | `b` | basis block | uniformity flag + basis descriptor(s), see *Basis descriptors* |
| 45 + 2`d` + 4 + `b` | `n * (d + 8)` | table entries | subspace table (see below) |
| 45 + 2`d` + 4 + `b` + `n * (d + 8)` | 4 | uint32 | CRC-32 (zlib) checksum of all preceding header bytes, verified on read |

Each **subspace table** entry is:

| size (bytes) | type | field |
|---|---|---|
| `d` | uint8 each | level vector of the subspace |
| 8 | uint64 | absolute byte offset of the subspace record from file start |

The table makes every record independently addressable, so a reader can
select subspaces by level (e.g. to interpolate only at relevant scales)
without scanning the file. Records are laid out contiguously after the
header, in table order.

### Subspace record

| size (bytes) | type | field |
|---|---|---|
| `4 * d` | uint32 each | extents (logical shape of the subspace) |
| 1 | uint8 | linearization order code: `C` = 0, `F` = 1, `ZC` = 2, `ZF` = 3 |
| 1 | uint8 | tensor kind: `EMPTY` = 0, `FULL` = 1, `LINEAR` = 2 |
| 1 | char | value dtype: numpy kind character (`f` float, `i` signed int, `u` unsigned int, ...) |
| 1 | uint8 | value dtype: item size in bytes (together e.g. `f8` = float64, `i1` = int8) |
| 2 | uint16 | precision bits |
| 2 | uint16 | padding bits |
| 1 | uint8 | compression (0 = none; reserved) |
| 4 | float32 | `quantization_scale` (reserved, see below) |
| 4 | int32 | `quantization_zero_point` (reserved) |
| 8 | uint64 | number of stored entries |
| 8 | uint64 | number of bytes in the data blob |
| 4 | uint32 | CRC-32 (zlib) checksum of the data blob, verified on read |
| (blob) | bytes | data blob, see below |

### Data blob

The blob content depends on the tensor kind:

- **`EMPTY` (0)**: no blob (zero bytes); every entry of the subspace is
  implicitly zero. This is deliberately the all-zero-bytes default: a
  zero-initialized record reads as "no data".
- **`FULL` (1)**: the complete linear value buffer, `prod(extents)` entries
  of the value dtype, in the subspace's linearization order.
- **`LINEAR` (2)**: the sorted linear indices of the stored entries, followed
  by the matching value buffer. Indices are stored in the **smallest unsigned
  integer type that can hold `prod(extents) - 1`** (uint8 up to 256 entries,
  uint16 up to 2^16, uint32 up to 2^32, else uint64); this type is derived
  from the extents and not stored explicitly. Entries not listed are
  implicitly zero.

Future kinds (e.g. interval/run-based sparsity) get new tensor-kind values.

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

### Linearization orders

The order code describes how the n-dimensional subspace is flattened into
the linear buffer (and what the `LINEAR` indices refer to):

- **`C` (0)**: row-major, last dimension varies fastest.
- **`F` (1)**: column-major, first dimension varies fastest.
- **`ZC` (2)** / **`ZF` (3)**: Z-order (Morton) curves; dimensions are visited
  round-robin starting with dimension 0 (`ZC`) or the last dimension (`ZF`),
  where a dimension drops out of the rotation once its extent is exhausted.
  For power-of-two extents this is classic bit interleaving; for arbitrary
  extents the curve bisects each dimension's remaining extent into
  ceil/floor halves, staying bijective without padding.

### Integrity and limits

- A reader must verify the magic string, reject unknown major/minor
  versions, verify the header CRC-32 before trusting the subspace table, and
  verify each record's CRC-32 before trusting its blob.
- The number of dimensions is a uint16: 1 to 65535. Levels are stored as
  single bytes. Extents are uint32 per dimension -- which effectively
  caps levels to 32 per dimension -- and byte counts are uint64.
- `precision_bits` is carried per subspace but not yet enforced as a storage
  width; values are stored at their dtype's width. `compression` is reserved
  and must currently be 0.

## License

The spght code is licensed under [Apache-2.0](LICENSES/Apache-2.0.txt);
documentation and configuration files are [CC-BY-4.0](LICENSES/CC-BY-4.0.txt)
or [CC0-1.0](LICENSES/CC0-1.0.txt) as marked. The repository follows the
[REUSE](https://reuse.software/) license specification.
