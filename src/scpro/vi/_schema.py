from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Sequence

import numpy as np

from scpro._compat import optional_import
from scpro._utils import as_dense


def from_combined_anndata(
    adata,
    *,
    modality_key: str = "feature_type",
    rna_values: Sequence[str] = ("rna", "Gene Expression", "RNA"),
    protein_values: Sequence[str] = ("protein", "ADT", "Antibody Capture", "Protein"),
    rna_name: str = "rna",
    protein_name: str = "protein",
):
    """Convert a combined RNA+protein AnnData object to MuData.

    Parameters are deliberately flexible because CITE-seq datasets often encode
    modalities as `rna/protein`, `Gene Expression/Antibody Capture`, or `ADT`.
    """
    md = optional_import("mudata", extra="vi")
    ad = optional_import("anndata", extra="vi")
    if modality_key not in adata.var:
        raise KeyError(f"adata.var must contain modality_key={modality_key!r}")
    values = adata.var[modality_key].astype(str)
    rna_mask = values.isin([str(v) for v in rna_values]).to_numpy()
    protein_mask = values.isin([str(v) for v in protein_values]).to_numpy()
    if rna_mask.sum() == 0:
        raise ValueError(f"No RNA features found using values {rna_values}")
    if protein_mask.sum() == 0:
        raise ValueError(f"No protein features found using values {protein_values}")
    rna = adata[:, rna_mask].copy()
    protein = adata[:, protein_mask].copy()
    return md.MuData({rna_name: rna, protein_name: protein})


def ensure_mudata(
    data,
    *,
    modality_key: str = "feature_type",
    rna_values: Sequence[str] = ("rna", "Gene Expression", "RNA"),
    protein_values: Sequence[str] = ("protein", "ADT", "Antibody Capture", "Protein"),
    rna_mod: str = "rna",
    protein_mod: str = "protein",
):
    """Normalize supported SCPRO-VI inputs to MuData."""
    md = optional_import("mudata", extra="vi")
    ad = optional_import("anndata", extra="vi")
    if isinstance(data, md.MuData):
        return data
    if isinstance(data, ad.AnnData):
        return from_combined_anndata(
            data,
            modality_key=modality_key,
            rna_values=rna_values,
            protein_values=protein_values,
            rna_name=rna_mod,
            protein_name=protein_mod,
        )
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.suffix == ".h5mu":
            return md.read_h5mu(path)
        if path.suffix == ".h5ad":
            return from_combined_anndata(
                ad.read_h5ad(path),
                modality_key=modality_key,
                rna_values=rna_values,
                protein_values=protein_values,
                rna_name=rna_mod,
                protein_name=protein_mod,
            )
    raise TypeError("SCPRO-VI expects MuData, combined AnnData, .h5mu path, or .h5ad path.")


def validate_mudata(mdata, *, rna_mod: str = "rna", protein_mod: str = "protein") -> None:
    if rna_mod not in mdata.mod:
        raise KeyError(f"MuData is missing RNA modality {rna_mod!r}")
    if protein_mod not in mdata.mod:
        raise KeyError(f"MuData is missing protein modality {protein_mod!r}")
    rna = mdata.mod[rna_mod]
    protein = mdata.mod[protein_mod]
    if rna.n_obs != protein.n_obs:
        raise ValueError("RNA and protein modalities must have the same number of cells.")
    if list(rna.obs_names) != list(protein.obs_names):
        raise ValueError("RNA and protein modalities must have matching cell order / obs_names.")


def get_feature_names(adata) -> np.ndarray:
    if "feature_name" in adata.var:
        return adata.var["feature_name"].astype(str).to_numpy()
    return adata.var_names.astype(str).to_numpy()
