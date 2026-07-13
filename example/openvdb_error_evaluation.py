# SPDX-FileCopyrightText: 2026 Theresa Pollinger
#
# SPDX-License-Identifier: Apache-2.0

"""Postprocessing: voxel-wise error between OpenVDB files.

Results are written to a CSV file in addition to the printed table.

Example:
    python3 openvdb_error_evaluation.py wdas_cloud/wdas_cloud_sixteenth.vdb \
        spght_0.001_wdas_cloud_sixteenth.vdb spght_0.01_wdas_cloud_sixteenth.vdb
"""

import argparse
import csv
from os.path import getsize, exists, splitext
import numpy as np

import openvdb as vdb
import spght


def read_dense(grid, bbox_min, shape) -> np.ndarray:
    """Copy a grid to a dense array covering the given index-space box."""
    dense = np.full(shape, grid.background, dtype=np.float64)
    grid.copyToArray(dense, ijk=bbox_min)
    return dense


def union_bounding_box(grids) -> tuple[tuple[int, ...], tuple[int, ...]]:
    boxes = [grid.evalActiveVoxelBoundingBox() for grid in grids]
    bbox_min = tuple(int(c) for c in np.min([b[0] for b in boxes], axis=0))
    bbox_max = tuple(int(c) for c in np.max([b[1] for b in boxes], axis=0))
    return bbox_min, bbox_max


def vdb_num_coefficients(grid) -> int:
    """Stored value count of a VDB grid: all voxels in allocated leaf nodes
    plus one value per active tile."""
    voxels_per_leaf = 2 ** (3 * grid.nodeLog2Dims()[-1])
    num_active_tiles = sum(1 for item in grid.citerOnValues() if item.count > 1)
    return grid.leafCount() * voxels_per_leaf + num_active_tiles


def sibling_spght_file(vdb_filename: str) -> str | None:
    """The .spght file the compressed .vdb was reconstructed from, if any."""
    spght_filename = splitext(vdb_filename)[0] + ".spght"
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
        description="Errors and file sizes of .vdb files w.r.t. a reference .vdb file."
    )
    parser.add_argument("reference", type=str, help="Path to the original .vdb file.")
    parser.add_argument(
        "compressed", type=str, nargs="+", help="Paths of .vdb files to compare."
    )
    parser.add_argument("--field", type=str, default="density")
    parser.add_argument(
        "--csv",
        type=str,
        default="error_evaluation.csv",
        help="Path of the CSV output file.",
    )
    args = parser.parse_args()

    reference_grid = vdb.read(args.reference, args.field)
    compared_grids = [vdb.read(f, args.field) for f in args.compressed]

    bbox_min, bbox_max = union_bounding_box([reference_grid] + compared_grids)
    shape = tuple(int(c) for c in np.asarray(bbox_max) + 1 - np.asarray(bbox_min))
    print(f"comparing on index box {bbox_min}..{bbox_max}, shape {shape}")

    reference = read_dense(reference_grid, bbox_min, shape)
    reference_l1 = np.abs(reference).sum()
    reference_l2 = np.sqrt(np.square(reference).sum())
    reference_lmax = np.abs(reference).max()
    print(
        f"reference: {args.reference} ({getsize(args.reference)} bytes, "
        f"{vdb_num_coefficients(reference_grid)} coefficients), "
        f"L1 norm {reference_l1:.6g}, L2 norm {reference_l2:.6g}, "
        f"Lmax norm {reference_lmax:.6g}"
    )

    fieldnames = [
        "file",
        "spght_bytes",
        "spght_coefficients",
        "vdb_bytes",
        "vdb_coefficients",
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
    for filename, grid in zip(args.compressed, compared_grids):
        diff = np.abs(read_dense(grid, bbox_min, shape) - reference)
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
                "vdb_bytes": getsize(filename),
                "vdb_coefficients": vdb_num_coefficients(grid),
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
        f"{'vdb MiB':>8} {'vdb coeffs':>10} "
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
            f"{row['vdb_bytes'] / 2**20:>8.2f} "
            f"{row['vdb_coefficients']:>10} "
            f"{row['l1_total']:>12.6g} {row['l1_mean']:>12.6g} "
            f"{row['l1_rel']:>10.4g} "
            f"{row['l2_total']:>12.6g} {row['l2_rms']:>12.6g} "
            f"{row['l2_rel']:>10.4g} "
            f"{row['lmax']:>10.4g} {row['lmax_rel']:>10.4g}",
            flush=True,
        )
