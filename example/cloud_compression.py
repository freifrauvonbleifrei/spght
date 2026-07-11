import argparse
import math
import numpy as np
from os.path import basename

import openvdb as vdb
from icecream import ic

from spght.compress import compress
from spght.hierarchize import hierarchize
from spght.interpolate import interpolate
from spght.data_structures import SparseGridHierarchicalTensors


def vdb_grid_extent(grid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bbox_min, bbox_max_exclusive, extent_per_dim) in index space."""
    bbox_min, bbox_max = grid.evalActiveVoxelBoundingBox()
    extent = np.array(bbox_max, dtype=np.int64) + 1 - np.array(bbox_min, dtype=np.int64)
    return bbox_min, bbox_max, extent


def levels_from_extent(extent: np.ndarray) -> np.ndarray:
    """Compute per-dimension level = ceil(log2(extent)) for power-of-2 padding."""
    return np.array([int(math.ceil(math.log2(max(e, 1)))) for e in extent])


def spght_to_vdb_grid(
    hierarchical_tensors: SparseGridHierarchicalTensors,
    bbox_min: np.ndarray,
    field_name: str = "density",
) -> vdb.FloatGrid:
    """Interpolate a sparse grid hierarchical tensor onto an OpenVDB grid."""
    # Compute the shape of the full tensor from the hierarchical representation
    full_shape = tuple(2**level for level in hierarchical_tensors.max_level)
    # compute the coordinates of the midpoints of the full tensor in index space

    # unit cube domain for interpolation
    unit_voxel_size = np.ones((3,), dtype=np.float32) / full_shape

    ii, jj, kk = np.meshgrid(
        np.arange(full_shape[0]),
        np.arange(full_shape[1]),
        np.arange(full_shape[2]),
        indexing="ij",
    )

    midpoints = np.stack(
        (
            (ii + 0.5) * unit_voxel_size[0],
            (jj + 0.5) * unit_voxel_size[1],
            (kk + 0.5) * unit_voxel_size[2],
        ),
        axis=-1,
    )
    nodal_values = interpolate(midpoints, hierarchical_tensors)

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
        help="Path to the output file where the compressed tensor will be saved.",
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
    nodal_values = np.full(shape, grid.background, dtype=np.float32)
    grid.copyToArray(nodal_values, ijk=bbox_min)

    # Hierarchize the tensor
    hierarchical_values = hierarchize(nodal_values)
    ic(len(hierarchical_values.subspaces))
    compressed_values = compress(
        hierarchical_values, epsilon=args.epsilon, only_whole_subspaces=True
    )
    ic(len(compressed_values.subspaces))

    # re-interpolate onto OpenVDB grid
    spght_openvdb_grid = spght_to_vdb_grid(
        compressed_values, bbox_min, field_name="density"
    )
    spght_openvdb_grid.transform = grid.transform  # preserve original transform
    spght_openvdb_grid.prune(tolerance=0.0)  # collapse uniform regions to save memory
    vdb.write(f"spght_{args.epsilon}_{basename(args.input_file)}", grids=[spght_openvdb_grid])

    # optional: invoke blender to render the compressed cloud
