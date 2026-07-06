import numpy as np

from scpro.vi._graphs import filter_weights, protein_similarity


def test_protein_similarity_small_numpy():
    x = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    weights = np.ones((2, 2), dtype=np.float32)
    sim = protein_similarity(x, weights, backend="numpy", block_size=1)
    assert sim.shape == (2, 2)
    assert np.isinf(sim[0, 0])
    assert sim[0, 1] >= 0
