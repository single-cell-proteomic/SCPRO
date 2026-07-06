from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from ._features import cluster_similarity


@dataclass
class ClusterMatching:
    cluster_relations: dict[str, list[str]]
    components: dict[str, list[str]]
    batch_relations: dict[str, list[tuple[str, str]]]
    unmatched_clusters: list[str]
    diff_map: dict[str, str]


def nearest_one_way(query: np.ndarray, target: np.ndarray) -> tuple[list[int], list[int]]:
    """One-way nearest-neighbor anchors matching the active paper pipeline."""
    if query.shape[0] == 0 or target.shape[0] == 0:
        return [], []
    idx = cKDTree(target).query(x=query, k=1, workers=-1)[1]
    return list(range(query.shape[0])), np.asarray(idx).tolist()


def _batch_token(cluster_id: str) -> tuple[str, ...]:
    # Original clusters are label_batchname; preserve support for underscores in batch names.
    return tuple(cluster_id.split("_")[1:])


def match_clusters(
    cluster_features: dict[str, dict[str, float]],
    *,
    top_k: int = 1,
    threshold: float | None = None,
) -> ClusterMatching:
    """Match clusters across batches through distinctive-feature similarity."""
    cluster_relations: dict[str, list[str]] = {}
    diff_map: dict[str, str] = {}

    for cluster, feats in cluster_features.items():
        scored: list[tuple[float, str]] = []
        most_diff: tuple[float, str] | None = None
        for other, other_feats in cluster_features.items():
            if cluster == other or _batch_token(cluster) == _batch_token(other):
                continue
            score = cluster_similarity(feats, other_feats)
            scored.append((score, other))
            if most_diff is None or score < most_diff[0]:
                most_diff = (score, other)

        scored.sort(key=lambda x: x[0], reverse=True)
        if threshold is not None:
            scored = [s for s in scored if s[0] >= threshold]
        if top_k is not None and top_k > 0:
            scored = scored[:top_k]
        cluster_relations[cluster] = [c for _, c in scored]
        if most_diff is not None:
            diff_map[cluster] = most_diff[1]

    unmatched = [c for c, rel in cluster_relations.items() if len(rel) == 0]
    edges: list[tuple[str, str]] = []
    for cluster, related in cluster_relations.items():
        for target in related:
            edges.append((cluster, target))

    graph = nx.Graph()
    graph.add_nodes_from(cluster_features.keys())
    graph.add_edges_from(edges)

    components: dict[str, list[str]] = {}
    batch_relations: dict[str, list[tuple[str, str]]] = {}
    for idx, component in enumerate(nx.connected_components(graph)):
        group = f"group_{idx}"
        components[group] = sorted(component)
        batch_relations[group] = []
        for cluster in components[group]:
            for target in cluster_relations.get(cluster, []):
                if target in component:
                    edge = (cluster, target)
                    rev = (target, cluster)
                    if rev not in batch_relations[group]:
                        batch_relations[group].append(edge)

    return ClusterMatching(
        cluster_relations=cluster_relations,
        components=components,
        batch_relations=batch_relations,
        unmatched_clusters=unmatched,
        diff_map=diff_map,
    )


def sample_negatives(obs_index: np.ndarray, n: int, *, rng: np.random.Generator) -> np.ndarray:
    if len(obs_index) == 0:
        raise ValueError("Cannot sample negatives from an empty candidate set.")
    return rng.choice(obs_index, size=n, replace=True)
