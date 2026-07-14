# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing: node-wise error between HDF5 files.

Results are written to a CSV file in addition to the printed table.

Example:
    python3 hdf5_error_evaluation.py hw2d_128.h5 \
        spght_0.001_hw2d_128.h5 spght_0.01_hw2d_128.h5
"""

import argparse
import csv
from os.path import getsize, exists, splitext
import numpy as np
import h5py

import spght


def read_field(filename: str, field: str, frame: int) -> np.ndarray:
    """One snapshot of the field; time-series datasets are indexed by frame."""
    with h5py.File(filename) as h5_file:
        dataset = h5_file[field]
        values = dataset[frame] if dataset.ndim == 3 else dataset[...]
        return values.astype(np.float64)


def sibling_spght_file(h5_filename: str) -> str | None:
    """The .spght file the compressed .h5 was reconstructed from, if any."""
    spght_filename = splitext(h5_filename)[0] + ".spght"
    return spght_filename if exists(spght_filename) else None


def spght_num_coefficients(spght_filename: str) -> int:
    """Stored value count of an .spght file: nnz for sparse subspaces, the
    full tensor size for dense ones."""

    tensors = spght.read(spght_filename)
    return sum(
        subspace.data.nnz if subspace.data.is_sparse else subspace.data.size
        for subspace in tensors.subspaces.values()
        if subspace.data is not None
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Errors and file sizes of .h5 files w.r.t. a reference .h5 file."
    )
    parser.add_argument("reference", type=str, help="Path to the original .h5 file.")
    parser.add_argument(
        "compressed", type=str, nargs="+", help="Paths of .h5 files to compare."
    )
    parser.add_argument("--field", type=str, default="density")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Time frame compared in time-series files; the last one by default.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="error_evaluation.csv",
        help="Path of the CSV output file.",
    )
    args = parser.parse_args()

    reference = read_field(args.reference, args.field, args.frame)
    reference_l1 = np.abs(reference).sum()
    reference_l2 = np.sqrt(np.square(reference).sum())
    reference_lmax = np.abs(reference).max()
    print(
        f"reference: {args.reference} ({getsize(args.reference)} bytes), "
        f"field {args.field} with shape {reference.shape}, "
        f"L1 norm {reference_l1:.6g}, L2 norm {reference_l2:.6g}, "
        f"Lmax norm {reference_lmax:.6g}"
    )

    fieldnames = [
        "file",
        "spght_bytes",
        "spght_coefficients",
        "h5_bytes",
        "h5_values",
        "l1_total",
        "l1_mean",
        "l1_rel",
        "l2_total",
        "l2_rms",
        "l2_rel",
        "lmax",
        "lmax_rel",
    ]
    rows = []
    for filename in args.compressed:
        compared = read_field(filename, args.field, args.frame)
        diff = np.abs(compared - reference)
        l1 = diff.sum()
        l2 = np.sqrt(np.square(diff).sum())
        spght_filename = sibling_spght_file(filename)
        rows.append(
            {
                "file": filename,
                "spght_bytes": getsize(spght_filename) if spght_filename else None,
                "spght_coefficients": (
                    spght_num_coefficients(spght_filename) if spght_filename else None
                ),
                "h5_bytes": getsize(filename),
                "h5_values": compared.size,
                "l1_total": l1,
                "l1_mean": diff.mean(),
                "l1_rel": l1 / reference_l1,
                "l2_total": l2,
                "l2_rms": np.sqrt(np.square(diff).mean()),
                "l2_rel": l2 / reference_l2,
                "lmax": diff.max(),
                "lmax_rel": diff.max() / reference_lmax,
            }
        )

    with open(args.csv, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.csv}")

    name_width = max(len(f) for f in args.compressed)
    header = (
        f"{'file':<{name_width}} {'spght MiB':>10} {'spght coeffs':>12} "
        f"{'h5 MiB':>8} {'h5 values':>10} "
        f"{'L1 total':>12} {'L1 mean':>12} {'L1 rel':>10} "
        f"{'L2 total':>12} {'L2 rms':>12} {'L2 rel':>10} "
        f"{'Lmax':>10} {'Lmax rel':>10}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        spght_mib = (
            f"{row['spght_bytes'] / 2**20:>10.2f}"
            if row["spght_bytes"] is not None
            else f"{'-':>10}"
        )
        spght_coeffs = (
            f"{row['spght_coefficients']:>12}"
            if row["spght_coefficients"] is not None
            else f"{'-':>12}"
        )
        print(
            f"{row['file']:<{name_width}} {spght_mib} {spght_coeffs} "
            f"{row['h5_bytes'] / 2**20:>8.2f} "
            f"{row['h5_values']:>10} "
            f"{row['l1_total']:>12.6g} {row['l1_mean']:>12.6g} "
            f"{row['l1_rel']:>10.4g} "
            f"{row['l2_total']:>12.6g} {row['l2_rms']:>12.6g} "
            f"{row['l2_rel']:>10.4g} "
            f"{row['lmax']:>10.4g} {row['lmax_rel']:>10.4g}",
            flush=True,
        )
