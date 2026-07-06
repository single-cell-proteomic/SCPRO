import numpy as np

from scpro.hi._features import cluster_similarity


def test_cluster_similarity_common_features():
    a = {"CD3": 0.5, "CD4": 0.5}
    b = {"CD3": 1.0, "CD8": 0.1}
    assert cluster_similarity(a, b) > 0


def test_cluster_similarity_no_common_features():
    assert cluster_similarity({"A": 1}, {"B": 1}) == 0
