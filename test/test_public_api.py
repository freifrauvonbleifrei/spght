import io

import numpy as np

import spght


def test_all_exports_resolve():
    for name in spght.__all__:
        assert getattr(spght, name) is not None


def test_public_api_end_to_end():
    # the whole pipeline is reachable from the top-level namespace
    nodal_values = np.random.default_rng(0).random((4, 4))
    tensors = spght.compress(spght.hierarchize(nodal_values), epsilon=0.0)
    assert isinstance(tensors, spght.SparseGridHierarchicalTensors)

    buffer = io.BytesIO()
    spght.write(tensors, buffer)
    buffer.seek(0)
    loaded = spght.read(buffer)

    coordinate = np.array([0.3, 0.6])
    assert np.isclose(
        spght.interpolate(coordinate, loaded),
        spght.interpolate(coordinate, tensors),
    )
