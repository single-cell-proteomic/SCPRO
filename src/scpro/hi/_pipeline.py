from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

from scpro._utils import as_dense

from ._features import FeatureSelector, select_distinctive_features
from ._matching import match_clusters, nearest_one_way, sample_negatives
from ._model import HIVAE, HIVAEConfig


@dataclass
class HIPipelineResult:
    adata: object
    cluster_features: dict[str, dict[str, float]]
    cluster_relations: dict[str, list[str]]
    components: dict[str, list[str]]
    cell_mappings: dict[str, list[tuple[str, str, str]]]


def _embedding(x: np.ndarray, *, n_components: int, random_state: int) -> np.ndarray:
    try:
        import umap

        n_comp = min(n_components, max(1, x.shape[0] - 2), max(1, x.shape[1] - 1))
        return umap.UMAP(n_components=n_comp, n_jobs=1, random_state=random_state).fit_transform(x)
    except ModuleNotFoundError:
        n_comp = min(n_components, max(1, x.shape[0] - 1), max(1, x.shape[1]))
        return PCA(n_components=n_comp, random_state=random_state).fit_transform(x)


def _majority(values) -> str:
    if len(values) == 0:
        return "unknown"
    return str(Counter(map(str, values)).most_common(1)[0][0])


def assign_clusters(
    hi_data,
    *,
    embedding_key: str = "X_scpro_hi_cluster",
    n_components: int = 5,
    bandwidth: float = 2.0,
    min_cluster_size: int = 50,
    random_state: int = 42,
) -> list[str]:
    """Cluster each dataset and return cluster IDs in concatenated order."""
    whole_cluster_list: list[str] = []
    for db in hi_data.dataset_list:
        x = as_dense(db.X)
        if embedding_key not in db.obsm:
            db.obsm[embedding_key] = _embedding(x, n_components=n_components, random_state=random_state)
        labels = MeanShift(bandwidth=bandwidth).fit(db.obsm[embedding_key]).labels_.astype(str)
        name = str(db.uns.get("name", "batch"))
        cluster_ids = np.asarray([f"{label}_{name}" for label in labels], dtype=object)

        clusters, counts = np.unique(cluster_ids, return_counts=True)
        if len(clusters) > 0:
            largest = clusters[np.argmax(counts)]
            for cluster, count in zip(clusters, counts):
                if count < min_cluster_size:
                    cluster_ids[cluster_ids == cluster] = largest

        # Robust handling of potential unclustered labels.
        noise_id = f"-1_{name}"
        if np.any(cluster_ids == noise_id) and np.any(cluster_ids != noise_id):
            clustered = np.where(cluster_ids != noise_id)[0]
            tree = cKDTree(db.obsm[embedding_key][clustered])
            for i in np.where(cluster_ids == noise_id)[0]:
                closest = clustered[tree.query(db.obsm[embedding_key][i])[1]]
                cluster_ids[i] = cluster_ids[closest]

        db.obs["cluster_id"] = cluster_ids
        whole_cluster_list.extend(cluster_ids.tolist())
    return whole_cluster_list


def build_cell_mappings(
    adata,
    *,
    batch_relations: dict[str, list[tuple[str, str]]],
    components: dict[str, list[str]],
    cluster_features: dict[str, dict[str, float]],
    diff_map: dict[str, str],
    extend: bool,
    random_state: int,
) -> tuple[dict[str, list[tuple[str, str, str]]], dict[str, list[str]]]:
    rng = np.random.default_rng(random_state)
    cell_mappings: dict[str, list[tuple[str, str, str]]] = {}
    negatives_for_prediction: dict[str, list[str]] = {}

    cluster_to_group = {}
    for group, clusters in components.items():
        for cluster in clusters:
            cluster_to_group[cluster] = group
    adata.obs["group_id"] = [cluster_to_group.get(c, "no-group") for c in adata.obs["cluster_id"]]

    for group, clusters in components.items():
        cell_mappings[group] = []
        negatives_for_prediction[group] = []
        plain_list = list(clusters)
        if not plain_list:
            continue

        for cluster_q in plain_list:
            query_cells = adata[adata.obs["cluster_id"].values == cluster_q]
            if query_cells.n_obs == 0:
                continue

            if len(plain_list) > 1:
                target_group = [x for x in plain_list if x != cluster_q and x.split("_")[1:] != cluster_q.split("_")[1:]]
                if not target_group:
                    target_group = [x for x in plain_list if x != cluster_q]
                target_cells = adata[np.isin(adata.obs["cluster_id"].values, target_group)]
                if target_cells.n_obs == 0:
                    target_cells = query_cells

                if extend:
                    query_features = set(cluster_features[cluster_q].keys())
                    target_features = set().union(*(set(cluster_features[t].keys()) for t in target_group if t in cluster_features))
                    features = list(query_features & target_features) or list(query_features)
                    m1, m2 = nearest_one_way(as_dense(query_cells[:, features].X), as_dense(target_cells[:, features].X))
                else:
                    m1, m2 = nearest_one_way(as_dense(query_cells.X), as_dense(target_cells.X))

                if m1:
                    q_ids = query_cells.obs_names.to_numpy()[m1]
                    t_ids = target_cells.obs_names.to_numpy()[m2]
                    neg_pool = adata[adata.obs["group_id"].values != group].obs_names.to_numpy()
                    if len(neg_pool) == 0:
                        neg_pool = adata[adata.obs["cluster_id"].values != cluster_q].obs_names.to_numpy()
                    if len(neg_pool) == 0:
                        neg_pool = query_cells.obs_names.to_numpy()
                    neg_ids = sample_negatives(neg_pool, len(q_ids), rng=rng)
                    cell_mappings[group].extend(list(zip(q_ids, t_ids, neg_ids)))

                pred_pool = adata[adata.obs["group_id"].values != group].obs_names.to_numpy()
                if len(pred_pool) == 0:
                    pred_pool = adata[adata.obs["cluster_id"].values != cluster_q].obs_names.to_numpy()
                if len(pred_pool) == 0:
                    pred_pool = query_cells.obs_names.to_numpy()
                negatives_for_prediction[group].extend(sample_negatives(pred_pool, query_cells.n_obs, rng=rng).tolist())
            else:
                neg_cluster = diff_map.get(cluster_q)
                if neg_cluster is not None:
                    neg_pool = adata[adata.obs["cluster_id"].values == neg_cluster].obs_names.to_numpy()
                else:
                    neg_pool = np.array([], dtype=object)
                if len(neg_pool) == 0:
                    neg_pool = adata[adata.obs["cluster_id"].values != cluster_q].obs_names.to_numpy()
                if len(neg_pool) == 0:
                    neg_pool = query_cells.obs_names.to_numpy()
                q_ids = query_cells.obs_names.to_numpy()
                neg_ids = sample_negatives(neg_pool, len(q_ids), rng=rng)
                cell_mappings[group].extend(list(zip(q_ids, q_ids, neg_ids)))
                negatives_for_prediction[group].extend(neg_ids.tolist())

    return cell_mappings, negatives_for_prediction


def run_pipeline(
    hi_data,
    *,
    n_features: int = 38,
    cluster_threshold: float | None = None,
    top_k_clusters: int = 1,
    extend: bool = False,
    result_key: str = "X_scpro_hi",
    random_state: int = 42,
    feature_selector: FeatureSelector = "keras_permutation",
    feature_epochs: int = 10,
    vae_epochs: int = 100,
    vae_batch_size: int = 16,
    umap_n_components: int = 5,
    mean_shift_bandwidth: float = 2.0,
    min_cluster_size: int = 50,
    label_key: str | None = None,
) -> HIPipelineResult:
    adata = hi_data.whole
    adata.obs["cluster_id"] = assign_clusters(
        hi_data,
        n_components=umap_n_components,
        bandwidth=mean_shift_bandwidth,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )

    n_features_eff = min(int(n_features), adata.n_vars)
    cluster_features: dict[str, dict[str, float]] = {}
    for db in hi_data.dataset_list:
        for cluster in db.obs["cluster_id"].unique():
            labels = [1 if x == cluster else 0 for x in db.obs["cluster_id"]]
            cluster_features[str(cluster)] = select_distinctive_features(
                db,
                labels,
                n_features_eff,
                selector=feature_selector,
                random_state=random_state,
                epochs=feature_epochs,
            )

    matching = match_clusters(cluster_features, top_k=top_k_clusters, threshold=cluster_threshold)
    cell_mappings, negatives_for_prediction = build_cell_mappings(
        adata,
        batch_relations=matching.batch_relations,
        components=matching.components,
        cluster_features=cluster_features,
        diff_map=matching.diff_map,
        extend=extend,
        random_state=random_state,
    )

    cc_features: dict[str, list[str]] = {}
    models: dict[str, HIVAE] = {}
    x_original = as_dense(adata.X)
    new_features = x_original.copy()

    for group in matching.components:
        labels = [1 if x == group else 0 for x in adata.obs["group_id"]]
        cc_features[group] = list(
            select_distinctive_features(
                adata,
                labels,
                n_features_eff,
                selector=feature_selector,
                random_state=random_state,
                epochs=feature_epochs,
            ).keys()
        )
        mappings = cell_mappings.get(group, [])
        if len(mappings) == 0:
            continue
        tuple_cell, tuple_pos, tuple_neg = zip(*mappings)
        train1 = as_dense(adata[list(tuple_cell), cc_features[group]].X)
        train2 = as_dense(adata[list(tuple_pos), cc_features[group]].X)
        train3 = as_dense(adata[list(tuple_neg), cc_features[group]].X)

        cfg = HIVAEConfig(
            input_size=len(cc_features[group]),
            dense_layer_size=len(cc_features[group]) * 3,
            latent_size=len(cc_features[group]),
            epochs=vae_epochs,
            batch_size=vae_batch_size,
        )
        model = HIVAE(cfg)
        model.train(train1, train2, train3)
        models[group] = model

    for group, features in cc_features.items():
        filtered = adata[adata.obs["group_id"].values == group]
        if filtered.n_obs == 0:
            continue
        i_cells = adata.obs_names.get_indexer(filtered.obs_names)
        current = new_features[i_cells].copy()
        if group in models:
            old = as_dense(filtered[:, features].X)
            neg_ids = negatives_for_prediction.get(group, [])
            if len(neg_ids) < filtered.n_obs:
                pool = adata.obs_names.to_numpy()
                rng = np.random.default_rng(random_state)
                neg_ids = rng.choice(pool, size=filtered.n_obs, replace=True).tolist()
            neg = as_dense(adata[neg_ids[: filtered.n_obs], features].X)
            corrected = models[group].predict(old, old, neg)
        else:
            corrected = as_dense(filtered[:, features].X)
        i_features = adata.var_names.get_indexer(features)
        current[:, i_features] = corrected
        new_features[i_cells] = current

    adata.obsm[result_key] = new_features
    adata.uns["scpro_hi"] = {
        "result_key": result_key,
        "n_features": n_features_eff,
        "cluster_threshold": cluster_threshold,
        "top_k_clusters": top_k_clusters,
        "feature_selector": feature_selector,
        "components": matching.components,
        "cluster_relations": matching.cluster_relations,
    }

    if label_key is not None and label_key in adata.obs:
        adata.uns["scpro_hi"]["label_key"] = label_key
        # Diagnostics can be added here without making labels a core requirement.

    return HIPipelineResult(
        adata=adata,
        cluster_features=cluster_features,
        cluster_relations=matching.cluster_relations,
        components=matching.components,
        cell_mappings=cell_mappings,
    )
