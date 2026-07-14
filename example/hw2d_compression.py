# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import replace
from os.path import basename, getsize, splitext

import h5py
import numpy as np

import spght


def cast_precision(
    tensors: spght.SparseGridHierarchicalTensors, precision_bits: int
) -> spght.SparseGridHierarchicalTensors:
    """Cast subspace coefficient values to the given float precision."""
    dtype = {16: np.float16, 32: np.float32, 64: np.float64}[precision_bits]
    for level, subspace in tensors.subspaces.items():
        data = subspace.data
        if data is not None and data.dtype != dtype:
            if data.is_sparse:
                data = spght.SparseTensor.from_linear(
                    data.linear_indices,
                    data.linear_values.astype(dtype),
                    data.shape,
                    order=data.order,
                )
            else:
                data = spght.DenseTensor(
                    data.linear_values.astype(dtype), data.shape, order=data.order
                )
        tensors.subspaces[level] = replace(
            subspace, data=data, precision_bits=precision_bits
        )
    return tensors


def spght_to_hdf5(
    hierarchical_tensors: spght.SparseGridHierarchicalTensors,
    output_file: str,
    field: str,
    dtype: np.dtype,
    attrs: dict,
) -> None:
    """Synthesize a sparse grid hierarchical tensor into an .h5 field."""
    nodal_values = spght.dehierarchize(hierarchical_tensors)

    with h5py.File(output_file, "w") as h5_file:
        h5_file.create_dataset(field, data=nodal_values.astype(dtype))
        for key, value in attrs.items():
            h5_file.attrs[key] = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress a field snapshot of a HW2D (Hasegawa-Wakatani) "
        ".h5 file to spght format, and output the compressed as .h5 again."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input .h5 file, as written by `python -m hw2d`.",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="density",
        choices=["density", "omega", "phi"],
        help="Name of the field to compress.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Time frame of the field to compress; the last one by default.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path of the .spght output file where the compressed tensor will "
        "be saved; derived from the input file name by default.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Threshold for compressing subspaces. Subspaces with values below this threshold will be removed.",
    )
    parser.add_argument(
        "--precision-bits",
        type=int,
        default=32,
        choices=[16, 32, 64],
        help="Float precision of the coefficient values in the .spght file.",
    )
    args = parser.parse_args()

    # Load one snapshot of the field from the .h5 input file
    with h5py.File(args.input_file) as h5_file:
        dataset = h5_file[args.field]
        frame = args.frame % dataset.shape[0]
        nodal_values = dataset[frame].astype(np.float64)
        field_dtype = dataset.dtype
        attrs = dict(h5_file.attrs)
        attrs["frame"] = frame
        if "frame_dt" in attrs:
            attrs["time"] = frame * attrs["frame_dt"]

    # Hierarchize the tensor: the HW2D fields are nodal values on a doubly
    # periodic 2^l x 2^l grid, a perfect fit for the periodic vertex-centered
    # CDF(2,2) basis
    wavelet = spght.cdf_2_2_basis(periodic=True)
    hierarchical_values = spght.hierarchize(nodal_values, wavelet=wavelet)

    def report(label: str, tensors) -> None:
        num_bytes = sum(s.num_bytes for s in tensors.subspaces.values())
        nnz = sum(s.data.nnz for s in tensors.subspaces.values() if s.data is not None)
        print(
            f"{label}: {len(tensors.subspaces)} subspaces, "
            f"{nnz} nonzero coefficients, {num_bytes / 2**20:.2f} MiB"
        )

    report("Previously", hierarchical_values)
    # partial compression: surviving coefficients per subspace are kept and
    # stored sparsely where that pays off
    compressed_values = spght.compress(hierarchical_values, epsilon=args.epsilon)
    compressed_values = cast_precision(compressed_values, args.precision_bits)
    report("After compression", compressed_values)

    # write the compressed tensor to a .spght file
    input_stem = splitext(basename(args.input_file))[0]
    spght_file = args.output_file or f"spght_{args.epsilon}_{input_stem}.spght"
    compressed_values.write(spght_file)
    print(f"Wrote {spght_file} ({getsize(spght_file) / 2**20:.2f} MiB)")

    # re-synthesize the field onto its full grid, from the file we just wrote
    loaded_values = spght.read(spght_file)
    spght_to_hdf5(
        loaded_values,
        f"spght_{args.epsilon}_{basename(args.input_file)}",
        args.field,
        field_dtype,
        attrs,
    )
