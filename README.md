# scpro

`scpro` is a clean Python package interface for two SCPRO methods:

- **SCPRO-HI**: horizontal integration of single-cell proteomic datasets through distinctive proteins in cell clusters.
- **SCPRO-VI**: multimodal RNA–protein integration with PPI-aware graph variational inference.

The package is intentionally **method-core only**. Research-era benchmarking scripts, alternative inactive matching functions, MOFA/Mowgli/totalVI comparison wrappers, notebook globals, hard-coded Google Drive paths, and import-time `pip install` side effects were removed from the clean package.

> Status: alpha / research package. The API is designed for reproducible method development and paper-code stabilization before public release.

---

## Design principles

`scpro` keeps the two methods under one package while using the right scientific data container for each method:

| Method | Internal object | Rationale |
|---|---:|---|
| `scpro.hi` | `AnnData` | SCPRO-HI integrates horizontally comparable single-cell proteomic datasets. Any supported external input is converted to one or more `AnnData` objects internally. |
| `scpro.vi` | `MuData` | SCPRO-VI is naturally multimodal: RNA and protein modalities are separate but share cells. Combined `AnnData` inputs can be converted to `MuData`. |

The package writes outputs into standard single-cell containers:

```text
SCPRO-HI: adata.obsm["X_scpro_hi"]
SCPRO-VI: mdata.obsm["X_scpro_vi"]
```

Graph and similarity matrices are stored in `.obsp`/`.uns` rather than non-standard `.obsm` slots.

---

## Installation

For local development:

```bash
git clone https://github.com/YOUR_USERNAME/scpro.git
cd scpro
pip install -e ".[hi,vi]"
```

Install only the module you need:

```bash
pip install -e ".[hi]"   # SCPRO-HI dependencies: AnnData/Scanpy/UMAP/TensorFlow
pip install -e ".[vi]"   # SCPRO-VI dependencies: AnnData/MuData/PyTorch/PyG
```

GPU acceleration for VI protein similarity is optional and should be installed explicitly for the matching CUDA stack, for example:

```bash
pip install -e ".[vi,gpu]"
```

---

## SCPRO-HI quick start

SCPRO-HI accepts an `AnnData`, a list of `AnnData` objects, or a directory of `.h5ad` files. Internally, all inputs are normalized into `AnnData` objects, concatenated by common features, clustered per batch, matched through distinctive features, and corrected by the HI-VAE model.

```python
import scpro as sp

adata = sp.hi.run(
    "data/protein_batches/",
    batch_key="batch_id",
    label_key="cell_type",      # optional, only for diagnostics
    n_features=38,
    result_key="X_scpro_hi",
    random_state=42,
)

adata.obsm["X_scpro_hi"]
```

Recommended downstream use:

```python
import scanpy as sc

sc.pp.neighbors(adata, use_rep="X_scpro_hi")
sc.tl.umap(adata)
sc.pl.umap(adata, color="cell_type")
```

### Important SCPRO-HI implementation notes

The package fixes several paper-script issues while preserving the active method logic:

- no class-level shared `dataset_list` state;
- no global `_adata` object;
- `cluster_s` is no longer hard-coded; use `label_key` for optional diagnostics;
- `n_features` is not silently overwritten by the full feature count;
- unselected features are preserved by initializing corrected output from the original `X` rather than an all-zero matrix;
- cluster matching can use `top_k_clusters` and/or `cluster_threshold` explicitly;
- negative sampling has guards for small or degenerate datasets;
- no mandatory debug plot or evaluation call in the core pipeline.

---

## SCPRO-VI quick start

SCPRO-VI is MuData-native. Provide a `MuData` with `rna` and `protein` modalities:

```python
import scpro as sp

mdata = sp.vi.from_combined_anndata(
    adata,
    modality_key="feature_type",
    rna_values=("rna", "Gene Expression"),
    protein_values=("protein", "ADT", "Antibody Capture"),
)

sp.vi.run(
    mdata,
    ppi_path="ppi_weights.csv",
    result_key="X_scpro_vi",
    latent_dim=100,
    hidden_dim=256,
    num_epochs=20,
    num_hvgs=1000,
    pretrained=True,
    random_state=42,
)

mdata.obsm["X_scpro_vi"]
```

The PPI table must contain:

```text
subs1, subs2, combined_score
```

### SCPRO-VI output locations

```text
mdata.obsm["X_scpro_vi"]
mdata.obsp["scpro_vi_similarity"]
mdata.mod["rna"].obsp["scpro_vi_similarity"]
mdata.mod["protein"].obsp["scpro_vi_similarity"]
mdata.uns["scpro_vi"]["rna_edges"]
mdata.uns["scpro_vi"]["protein_edges"]
mdata.uns["scpro_vi"]["params"]
```

### Important SCPRO-VI implementation notes

The package removes import-time dependency installation and hard-coded PPI paths. Protein similarity can run in a memory-aware blockwise mode while preserving the original PPI-weighted outer-product distance definition.

Inactive research alternatives are not part of the core package, including backup matching functions, MOFA/Mowgli/totalVI wrappers, old VAE baselines, SCIB benchmarking utilities, notebook processing helpers, and Google Drive-specific file logic.

---

## Minimal API

```python
import scpro as sp

# Horizontal proteomics integration
adata_hi = sp.hi.run(adatas_or_path, result_key="X_scpro_hi")

# Multimodal RNA/protein integration
mdata = sp.vi.from_combined_anndata(adata)
sp.vi.run(mdata, ppi_path="ppi_weights.csv", result_key="X_scpro_vi")
```

---

## Reproducibility

Both modules expose `random_state`. The package sets NumPy, Python, TensorFlow, and/or PyTorch seeds where applicable. Full bitwise reproducibility may still depend on GPU kernels and third-party backend settings.

---

## Citation

If you use **SCPRO-HI**, please cite:

```bibtex
@article{Koca2024SCPROHI,
  title={Integration of single-cell proteomic datasets through distinctive proteins in cell clusters},
  author={Koca, Mehmet Burak and Sevilgen, Fatih Erdogan},
  journal={Proteomics},
  volume={24},
  pages={2300282},
  year={2024},
  doi={10.1002/pmic.202300282}
}
```

If you use **SCPRO-VI**, please cite:

```bibtex
@article{Koca2026SCPROVI,
  title={Explainable graph learning for multimodal single-cell data integration},
  author={Koca, Mehmet Burak and Sevilgen, Fatih Erdogan},
  journal={BMC Bioinformatics},
  year={2026},
  doi={10.1186/s12859-026-06413-3}
}
```

---

## License

Add the final project license before publishing to PyPI or GitHub releases.
