from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scpro._utils import set_random_seed

from ._data import HIData
from ._features import FeatureSelector
from ._pipeline import run_pipeline


@dataclass
class HIConfig:
    batch_key: str = "batch_id"
    label_key: str | None = None
    n_features: int = 38
    cluster_threshold: float | None = None
    top_k_clusters: int = 1
    extend: bool = False
    result_key: str = "X_scpro_hi"
    random_state: int = 42
    feature_selector: FeatureSelector = "keras_permutation"
    feature_epochs: int = 10
    vae_epochs: int = 100
    vae_batch_size: int = 16
    umap_n_components: int = 5
    mean_shift_bandwidth: float = 2.0
    min_cluster_size: int = 50
    l2_normalize: bool = True
    copy: bool = False


def run(
    data,
    *,
    batch_key: str = "batch_id",
    label_key: str | None = None,
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
    l2_normalize: bool = True,
    copy: bool = False,
):
    """Run SCPRO-HI and return the integrated AnnData object.

    Parameters
    ----------
    data
        AnnData, list of AnnData, `.h5ad` file path, or directory containing `.h5ad` files.
    batch_key
        Batch/dataset key. If absent, it is created from file names or `uns['name']`.
    label_key
        Optional label key for diagnostics only. The core method does not require labels.
    n_features
        Number of distinctive features used per cluster/group. Unlike the original
        script, this value is not silently overwritten by the total feature count.
    result_key
        Key in `adata.obsm` where corrected measurements are stored.
    copy
        Accepted for API clarity; inputs are copied during normalization/construction.
    """
    del copy  # inputs are normalized to a safe HIData copy regardless.
    set_random_seed(random_state)
    hi_data = HIData.from_input(data, batch_key=batch_key, l2_normalize=l2_normalize)
    result = run_pipeline(
        hi_data,
        n_features=n_features,
        cluster_threshold=cluster_threshold,
        top_k_clusters=top_k_clusters,
        extend=extend,
        result_key=result_key,
        random_state=random_state,
        feature_selector=feature_selector,
        feature_epochs=feature_epochs,
        vae_epochs=vae_epochs,
        vae_batch_size=vae_batch_size,
        umap_n_components=umap_n_components,
        mean_shift_bandwidth=mean_shift_bandwidth,
        min_cluster_size=min_cluster_size,
        label_key=label_key,
    )
    return result.adata
