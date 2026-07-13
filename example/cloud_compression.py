import argparse
import math
import numpy as np
from os.path import basename, getsize, splitext

import openvdb as vdb

import spght


def vdb_grid_extent(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bbox_min, bbox_max_exclusive, extent_per_dim) in index space."""
    bbox_min, bbox_max = grid.evalActiveVoxelBoundingBox()
    extent = np.array(bbox_max, dtype=np.int64) + 1 - np.array(bbox_min, dtype=np.int64)
    return bbox_min, bbox_max, extent


def levels_from_extent(extent: np.ndarray) -> np.ndarray:
    """Compute per-dimension level = ceil(log2(extent)) for power-of-2 padding."""
    return np.array([int(math.ceil(math.log2(max(e, 1)))) for e in extent])


def spght_to_vdb_grid(
    hierarchical_tensors: spght.SparseGridHierarchicalTensors,
    bbox_min: np.ndarray,
    field_name: str = "density",
) -> vdb.FloatGrid:
    """Interpolate a sparse grid hierarchical tensor onto an OpenVDB grid."""
    # Cell-midpoint coordinates of the full tensor in the [0, 1]^d unit domain.
    midpoints = spght.midpoint_coordinates_from_level(hierarchical_tensors.max_level)
    nodal_values = spght.interpolate(midpoints, hierarchical_tensors)

    # Create a new OpenVDB grid and copy the interpolated values into it
    grid = vdb.FloatGrid()
    grid.copyFromArray(nodal_values, ijk=bbox_min)
    grid.name = field_name

    return grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compress a vdb file to spght format, and output the compressed as vdb again."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input file containing the hierarchical tensor.",
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
    args = parser.parse_args()

    # Load the cloud from .vdb input file
    grid = vdb.read(args.input_file, "density")

    # unfurl cloud to a full-scale tensor
    bbox_min, bbox_max, tight_shape = vdb_grid_extent(grid)
    max_level = levels_from_extent(tight_shape)
    shape = 2**max_level
    nodal_values = np.full(shape, grid.background, dtype=np.float64)
    grid.copyToArray(nodal_values, ijk=bbox_min)

    # Hierarchize the tensor
    hierarchical_values = spght.hierarchize(nodal_values)

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
    report("After compression", compressed_values)

    # write the compressed tensor to a .spght file
    input_stem = splitext(basename(args.input_file))[0]
    spght_file = args.output_file or f"spght_{args.epsilon}_{input_stem}.spght"
    compressed_values.write(spght_file)
    print(f"Wrote {spght_file} ({getsize(spght_file) / 2**20:.2f} MiB)")

    # re-interpolate onto OpenVDB grid, from the file we just wrote
    loaded_values = spght.read(spght_file)
    spght_openvdb_grid = spght_to_vdb_grid(
        loaded_values, bbox_min, field_name="density"
    )
    spght_openvdb_grid.transform = grid.transform  # preserve original transform
    spght_openvdb_grid.prune(tolerance=0.0)  # collapse uniform regions to save memory
    vdb.write(
        f"spght_{args.epsilon}_{basename(args.input_file)}", grids=[spght_openvdb_grid]
    )

    # optional: invoke blender to render the compressed cloud
