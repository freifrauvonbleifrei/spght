
import numpy as np

from spght.compress import compress
from spght.hierarchize import hierarchize


def test_compress_small_checker():
    nodal_values = np.array([[1.0, 0.0], [0.0, 1.0]])
    hierarchical_values = hierarchize(nodal_values)
    compressed_values = (
        compress(hierarchical_values, only_whole_subspaces=True)
    )
    assert compressed_values.dimensions == 2
    assert compressed_values.max_level == (1, 1)
    assert len(compressed_values.subspaces) == 2
    assert compressed_values.subspaces[(0, 0)].values == [0.5]
    assert compressed_values.subspaces[(1, 1)].values == [0.5]
