from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

from scpro._utils import as_dense

from ._schema import get_feature_names, validate_mudata

Backend = Literal["auto", "numpy", "cupy"]


def read_ppi(ppi=None, ppi_path: str | Path | None = None) -> pd.DataFrame | None:
    if ppi is not None and ppi_path is not None:
        raise ValueError("Provide either ppi or ppi_path, not both.")
    if ppi is None and ppi_path is None:
        return None
    if ppi is not None:
        df = ppi.copy() if isinstance(ppi, pd.DataFrame) else pd.DataFrame(ppi)
    else:
        df = pd.read_csv(ppi_path)
    required = {"subs1", "subs2", "combined_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"PPI table is missing required columns: {sorted(missing)}")
    return df


def filter_weights(weights: pd.DataFrame | None, protein_names: np.ndarray, feature_count: int) -> np.ndarray:
    """Filter PPI weights to the protein features in the dataset."""
    if weights is None:
        return np.ones((feature_count, feature_count), dtype=np.float32)
    names = np.asarray(protein_names).astype(str)
    filtered = weights[weights["subs1"].astype(str).isin(names) & weights["subs2"].astype(str).isin(names)]
    matrix = np.zeros((feature_count, feature_count), dtype=np.float32)
    if filtered.shape[0] == 0:
        matrix[:] = 1.0
        return matrix
    name_to_idx: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        name_to_idx.setdefault(name, []).append(i)
    for _, row in filtered.iterrows():
        score = float(row["combined_score"])
        for i in name_to_idx.get(str(row["subs1"]), []):
            for j in name_to_idx.get(str(row["subs2"]), []):
                if i != j:
                    matrix[min(i, j), max(i, j)] = score
                    matrix[max(i, j), min(i, j)] = score
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _resolve_array_backend(backend: Backend):
    if backend == "numpy":
        return np
    if backend in {"auto", "cupy"}:
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError:
            if backend == "cupy":
                raise
            return np
    raise ValueError(f"Unknown backend: {backend}")


def protein_similarity(
    x,
    weights: np.ndarray,
    *,
    backend: Backend = "auto",
    block_size: int = 256,
    eps: float = 1e-12,
) -> np.ndarray:
    """PPI-weighted protein distance used by SCPRO-VI.

    This is mathematically equivalent to the active paper-code distance over
    PPI-weighted outer products, but it avoids allocating the full
    n_cells x n_proteins x n_proteins tensor.
    """
    xp = _resolve_array_backend(backend)
    x_np = as_dense(x, dtype=np.float32)
    n_cells, _ = x_np.shape
    weights_np = np.asarray(weights, dtype=np.float32)
    mask = weights_np > 0
    if mask.sum() == 0:
        mask = np.ones_like(weights_np, dtype=bool)
    src, dst = np.where(mask)
    w = weights_np[src, dst]

    x_backend = xp.asarray(x_np)
    w_backend = xp.asarray(w)
    # Flatten only biologically considered PPI entries instead of storing all outer products.
    ppi_features = (x_backend[:, src] * x_backend[:, dst]) * w_backend
    norms = xp.sum(x_backend, axis=1) + eps
    out = xp.zeros((n_cells, n_cells), dtype=xp.float32)

    for i0 in range(0, n_cells, block_size):
        i1 = min(i0 + block_size, n_cells)
        a = ppi_features[i0:i1]
        for j0 in range(i0, n_cells, block_size):
            j1 = min(j0 + block_size, n_cells)
            b = ppi_features[j0:j1]
            diff = xp.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
            denom = norms[i0:i1, None] * norms[j0:j1][None, :]
            block = diff / denom
            out[i0:i1, j0:j1] = block
            if j0 != i0:
                out[j0:j1, i0:i1] = block.T

    if xp is not np:
        out_np = xp.asnumpy(out)
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    else:
        out_np = np.asarray(out)
    finite = np.isfinite(out_np)
    if finite.any():
        out_np[~finite] = np.max(out_np[finite])
    else:
        out_np[:] = 0
    np.fill_diagonal(out_np, np.inf)
    return out_np


def knn_neighbors(similarities: np.ndarray, k: int) -> list[np.ndarray]:
    k = min(int(k), similarities.shape[0] - 1)
    if k < 1:
        return [np.array([], dtype=int) for _ in range(similarities.shape[0])]
    out = []
    for i in range(similarities.shape[0]):
        row = similarities[i].copy()
        row[i] = np.inf
        out.append(np.argsort(row)[:k])
    return out


def positive_edges(distances: np.ndarray, k: int, threshold: float) -> tuple[list[tuple[int, int]], np.ndarray]:
    adj = np.zeros((distances.shape[0], distances.shape[0]), dtype=np.float32)
    edges: list[tuple[int, int]] = []
    for i in range(distances.shape[0]):
        row = distances[i]
        below = np.where(row < threshold)[0]
        if below.size > 0:
            selected = below[np.argsort(row[below])]
            selected = selected[: min(k, selected.size)]
        else:
            selected = np.argsort(row)[:1]
        for j in selected:
            edges.append((int(i), int(j)))
        adj[i, selected] = 1.0
    np.fill_diagonal(adj, 1.0)
    return edges, adj


def cutoff_threshold(dist: np.ndarray, order: np.ndarray, *, bin_size: int = 50) -> float:
    n_bins = max(1, len(dist) // bin_size)
    for j in range(n_bins):
        selected = dist[order[: (j + 1) * bin_size]]
        mean = np.mean(selected)
        std = np.std(selected)
        if std > 0.20 * mean:
            if j != 0:
                return float(dist[order[j * bin_size]])
            return float(dist[order[min(2, len(order) - 1)]])
    return float(dist[order[len(dist) // 2]])


def normalize_by_local_neighbors(distances: np.ndarray, *, k: int) -> np.ndarray:
    distances = distances.astype(np.float32, copy=True)
    neighbors = knn_neighbors(distances, k)
    denom = np.asarray([
        np.mean(row[cols]) if len(cols) else 1.0 for row, cols in zip(distances, neighbors)
    ])
    denom[denom == 0] = 1.0
    distances = distances / denom[:, None]
    np.fill_diagonal(distances, np.inf)
    return distances


def build_graphs(
    mdata,
    *,
    ppi=None,
    ppi_path: str | Path | None = None,
    rna_mod: str = "rna",
    protein_mod: str = "protein",
    rna_rep: str | None = "embeddings",
    ppi_backend: Backend = "auto",
    protein_block_size: int = 256,
    rna_k: int = 40,
    protein_k: int = 100,
    edge_threshold: float = 1.0,
    key: str = "scpro_vi",
) -> None:
    """Build RNA, protein, and joint graphs for SCPRO-VI."""
    validate_mudata(mdata, rna_mod=rna_mod, protein_mod=protein_mod)
    rna = mdata.mod[rna_mod]
    protein = mdata.mod[protein_mod]
    ppi_df = read_ppi(ppi=ppi, ppi_path=ppi_path)

    if rna_rep is not None and rna_rep in rna.obsm:
        rna_x = rna.obsm[rna_rep]
    else:
        rna_x = as_dense(rna.X)
    rna_dist = pairwise_distances(rna_x, metric="cosine")
    rna_dist = normalize_by_local_neighbors(rna_dist, k=rna_k)

    protein_names = get_feature_names(protein)
    weights = filter_weights(ppi_df, protein_names, protein.n_vars)
    prot_dist = protein_similarity(
        protein.X,
        weights,
        backend=ppi_backend,
        block_size=protein_block_size,
    )
    prot_dist = normalize_by_local_neighbors(prot_dist, k=protein_k)

    joint = prot_dist * rna_dist
    finite = np.isfinite(joint)
    if finite.any():
        min_val = joint[finite].min()
        max_val = joint[finite].max()
        if max_val > min_val:
            joint = (joint - min_val) / (max_val - min_val)
    orders = np.argsort(joint, axis=1)
    np.fill_diagonal(joint, 0)
    thresholds = np.asarray([cutoff_threshold(joint[i], orders[i]) for i in range(joint.shape[0])])
    keep = joint < thresholds[:, None]
    filtered = joint * keep
    filtered += (~keep).astype(np.float32)
    np.fill_diagonal(filtered, 0)
    joint = np.minimum(filtered, filtered.T)

    prot_edges, prot_adj = positive_edges(prot_dist, protein_k, edge_threshold)
    rna_edges, rna_adj = positive_edges(rna_dist, protein_k, edge_threshold)

    rna.obsp[f"{key}_similarity"] = rna_dist
    protein.obsp[f"{key}_similarity"] = prot_dist
    rna.obsp[f"{key}_adjacency"] = rna_adj
    protein.obsp[f"{key}_adjacency"] = prot_adj
    mdata.obsp[f"{key}_similarity"] = joint
    mdata.uns.setdefault(key, {})
    mdata.uns[key]["rna_edges"] = np.asarray(rna_edges, dtype=np.int64).T if rna_edges else np.empty((2, 0), dtype=np.int64)
    mdata.uns[key]["protein_edges"] = np.asarray(prot_edges, dtype=np.int64).T if prot_edges else np.empty((2, 0), dtype=np.int64)
    mdata.uns[key]["ppi_edges_considered"] = int((weights > 0).sum())
