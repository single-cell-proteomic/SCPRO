from __future__ import annotations

from dataclasses import dataclass

from scpro.io import concat_common_features, ensure_anndata_list
from scpro._utils import as_dense


@dataclass
class HIData:
    """Container for SCPRO-HI inputs.

    Unlike the original paper script, this container uses instance attributes only;
    repeated runs cannot leak datasets through a class-level `dataset_list`.
    """

    dataset_list: list
    whole: object

    @classmethod
    def from_input(cls, data, *, batch_key: str = "batch_id", l2_normalize: bool = True) -> "HIData":
        adatas = ensure_anndata_list(data, batch_key=batch_key)
        for adata in adatas:
            adata.X = as_dense(adata.X)
            if l2_normalize:
                from sklearn.preprocessing import Normalizer

                adata.X = Normalizer(norm="l2").fit_transform(adata.X)
        whole, adatas = concat_common_features(adatas, batch_key=batch_key)
        return cls(dataset_list=adatas, whole=whole)
