from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ._compat import optional_import
from ._utils import as_dense


def read_h5ad(path: str | Path):
    """Read an `.h5ad` file using AnnData."""
    ad = optional_import("anndata", extra="hi")
    return ad.read_h5ad(path)


def read_h5ad_directory(path: str | Path, *, batch_key: str = "batch_id") -> list:
    """Read all `.h5ad` files in a directory as AnnData objects.

    The file stem is written to `obs[batch_key]` if the key is not already present.
    """
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Expected an h5ad directory, got: {path}")
    adatas = []
    for file in sorted(path.glob("*.h5ad")):
        adata = read_h5ad(file)
        if batch_key not in adata.obs:
            adata.obs[batch_key] = file.stem
        adata.uns["name"] = adata.uns.get("name", file.stem)
        adatas.append(adata)
    if not adatas:
        raise ValueError(f"No .h5ad files found in {path}")
    return adatas


def ensure_anndata_list(data, *, batch_key: str = "batch_id") -> list:
    """Normalize supported SCPRO-HI inputs to a list of AnnData objects."""
    ad = optional_import("anndata", extra="hi")
    AnnData = ad.AnnData

    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.is_dir():
            return read_h5ad_directory(p, batch_key=batch_key)
        if p.suffix == ".h5ad":
            adata = read_h5ad(p)
            if batch_key not in adata.obs:
                adata.obs[batch_key] = p.stem
            return [adata]
        raise ValueError("Path input must be an .h5ad file or a directory containing .h5ad files.")

    if isinstance(data, AnnData):
        if batch_key in data.obs:
            adatas = []
            for batch in data.obs[batch_key].astype(str).unique():
                subset = data[data.obs[batch_key].astype(str).values == str(batch)].copy()
                subset.uns["name"] = str(batch)
                adatas.append(subset)
            return adatas
        copy = data.copy()
        copy.obs[batch_key] = copy.uns.get("name", "batch_0")
        copy.uns["name"] = copy.uns.get("name", "batch_0")
        return [copy]

    if isinstance(data, Sequence):
        out = []
        for idx, item in enumerate(data):
            if not isinstance(item, AnnData):
                raise TypeError("All sequence items must be AnnData objects.")
            adata = item.copy()
            name = str(adata.uns.get("name", f"batch_{idx}"))
            adata.uns["name"] = name
            if batch_key not in adata.obs:
                adata.obs[batch_key] = name
            out.append(adata)
        if not out:
            raise ValueError("At least one AnnData object is required.")
        return out

    raise TypeError("Unsupported input. Provide AnnData, list[AnnData], .h5ad path, or .h5ad directory.")


def concat_common_features(adatas: Iterable, *, batch_key: str = "batch_id"):
    """Concatenate AnnData objects using the common feature intersection."""
    ad = optional_import("anndata", extra="hi")
    adatas = [a.copy() for a in adatas]
    for adata in adatas:
        adata.X = as_dense(adata.X)
    whole = ad.concat(adatas, join="inner", label=None)
    whole.obs_names_make_unique()
    whole.uns["name"] = "SCPRO-HI"
    # make sure each per-batch object has exactly the same features as the concat object
    adatas = [a[:, whole.var_names].copy() for a in adatas]
    return whole, adatas
