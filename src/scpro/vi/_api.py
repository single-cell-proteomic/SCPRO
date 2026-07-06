from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from scpro._utils import set_random_seed

from ._schema import ensure_mudata, from_combined_anndata


@dataclass
class VIConfig:
    latent_dim: int = 100
    hidden_dim: int = 256
    num_neighbors: tuple[int, ...] = (15,)
    num_epochs: int = 20
    num_hvgs: int = 1000
    pretrained: bool = True
    use_embeddings: bool = True
    device: str | None = None


def run(
    data,
    *,
    ppi=None,
    ppi_path: str | Path | None = None,
    rna_mod: str = "rna",
    protein_mod: str = "protein",
    modality_key: str = "feature_type",
    rna_values: Sequence[str] = ("rna", "Gene Expression", "RNA"),
    protein_values: Sequence[str] = ("protein", "ADT", "Antibody Capture", "Protein"),
    result_key: str = "X_scpro_vi",
    graph_key: str = "scpro_vi",
    latent_dim: int = 100,
    hidden_dim: int = 256,
    num_neighbors: Sequence[int] = (15,),
    num_epochs: int = 20,
    num_hvgs: int = 1000,
    pretrained: bool = True,
    use_embeddings: bool = True,
    random_state: int = 42,
    device: str | None = None,
    ppi_backend: str = "auto",
    protein_block_size: int = 256,
):
    """Run SCPRO-VI and return a MuData object.

    Parameters
    ----------
    data
        MuData, combined AnnData, `.h5mu` path, or combined `.h5ad` path.
    ppi / ppi_path
        PPI table as DataFrame-like object or CSV path. The table must contain
        `subs1`, `subs2`, and `combined_score`.
    result_key
        Key in `mdata.obsm` where the integrated embedding is stored.
    """
    set_random_seed(random_state)
    mdata = ensure_mudata(
        data,
        modality_key=modality_key,
        rna_values=rna_values,
        protein_values=protein_values,
        rna_mod=rna_mod,
        protein_mod=protein_mod,
    )
    from ._train import TrainingConfig, train_vi

    cfg = TrainingConfig(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_neighbors=tuple(num_neighbors),
        num_epochs=num_epochs,
        num_hvgs=num_hvgs,
        pretrained=pretrained,
        use_embeddings=use_embeddings,
        device=device,
    )
    mdata = train_vi(
        mdata,
        ppi=ppi,
        ppi_path=ppi_path,
        rna_mod=rna_mod,
        protein_mod=protein_mod,
        result_key=result_key,
        graph_key=graph_key,
        config=cfg,
        ppi_backend=ppi_backend,
        protein_block_size=protein_block_size,
    )
    mdata.uns.setdefault(graph_key, {})
    mdata.uns[graph_key]["params"] = {
        **asdict(cfg),
        "random_state": random_state,
        "ppi_backend": ppi_backend,
        "protein_block_size": protein_block_size,
    }
    return mdata
