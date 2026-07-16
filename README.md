# SCPRO

`scpro` is a Python package for single-cell proteomics integration. It provides a unified interface for two complementary methods:

- **SCPRO-HI** — horizontal integration of multiple single-cell proteomic datasets using distinctive proteins identified across cell clusters.
- **SCPRO-VI** — multimodal RNA–protein integration using PPI-aware graph variational inference.

## Installation

Clone the repository and install both modules:

```bash
git clone https://github.com/single-cell-proteomic/SCPRO.git
cd SCPRO
pip install -e ".[hi,vi]"
```

To install only one module:

```bash
# SCPRO-HI
pip install -e ".[hi]"

# SCPRO-VI
pip install -e ".[vi]"
```

Optional GPU dependencies for SCPRO-VI protein-similarity calculations can be installed with:

```bash
pip install -e ".[vi,gpu]"
```

GPU packages must be compatible with the CUDA version available on the system.

## SCPRO-HI

SCPRO-HI performs horizontal integration of single-cell proteomic datasets.

It accepts:

- an `AnnData` object;
- a list of `AnnData` objects; or
- a directory containing `.h5ad` files.

The datasets are aligned using their common protein features, clustered within each batch, matched through distinctive proteins, and corrected using the SCPRO-HI variational autoencoder.

### Quick start

```python
import scpro as sp

adata = sp.hi.run(
    "data/protein_batches/",
    batch_key="batch_id",
    label_key="cell_type",
    n_features=38,
    result_key="X_scpro_hi",
    random_state=42,
)
```

The integrated representation is stored in:

```python
adata.obsm["X_scpro_hi"]
```

The `label_key` parameter is optional and is used only for diagnostic analyses.

SCPRO-HI can also be run directly on an `AnnData` object or a list of `AnnData` objects:

```python
adata = sp.hi.run(
    adata_list,
    batch_key="batch_id",
    result_key="X_scpro_hi",
)
```

## SCPRO-VI

SCPRO-VI performs multimodal integration of paired RNA and protein measurements using graph variational inference.

The method is MuData-native and expects a `MuData` object containing:

- an `rna` modality;
- a `protein` modality.

### Converting a combined AnnData object

A combined `AnnData` object can be converted into the required `MuData` representation:

```python
import scpro as sp

mdata = sp.vi.from_combined_anndata(
    adata,
    modality_key="feature_type",
    rna_values=("rna", "Gene Expression"),
    protein_values=("protein", "ADT", "Antibody Capture"),
)
```

### Running SCPRO-VI

```python
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
```

The integrated representation is stored in:

```python
mdata.obsm["X_scpro_vi"]
```

### PPI input format

The PPI input file must contain the following columns:

```text
subs1,subs2,combined_score
```

Each row represents an interaction between two proteins and its corresponding confidence score.

Example:

```text
subs1,subs2,combined_score
CD3D,CD3E,0.998
CD3E,CD247,0.975
CD4,LCK,0.941
```

### SCPRO-VI outputs

SCPRO-VI stores the integrated representation, similarity graphs, graph edges, and model parameters in the input `MuData` object:

```python
# Integrated latent representation
mdata.obsm["X_scpro_vi"]

# Joint similarity graph
mdata.obsp["scpro_vi_similarity"]

# Modality-specific similarity graphs
mdata.mod["rna"].obsp["scpro_vi_similarity"]
mdata.mod["protein"].obsp["scpro_vi_similarity"]

# Graph edges
mdata.uns["scpro_vi"]["rna_edges"]
mdata.uns["scpro_vi"]["protein_edges"]

# Run parameters
mdata.uns["scpro_vi"]["params"]
```

## Citation

### SCPRO-HI

When using SCPRO-HI, please cite:

```bibtex
@article{Koca2024SCPROHI,
  title   = {Integration of single-cell proteomic datasets through distinctive proteins in cell clusters},
  author  = {Koca, Mehmet Burak and Sevilgen, Fatih Erdogan},
  journal = {Proteomics},
  volume  = {24},
  pages   = {2300282},
  year    = {2024},
  doi     = {10.1002/pmic.202300282}
}
```

### SCPRO-VI

When using SCPRO-VI, please cite:

```bibtex
@article{Koca2026SCPROVI,
  title   = {Explainable graph learning for multimodal single-cell data integration},
  author  = {Koca, Mehmet Burak and Sevilgen, Fatih Erdogan},
  journal = {BMC Bioinformatics},
  year    = {2026},
  doi     = {10.1186/s12859-026-06413-3}
}
```
