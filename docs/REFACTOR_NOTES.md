# Refactor notes

This package intentionally keeps only the active method cores from the supplied SCPRO-HI and SCPRO-VI projects.

## Removed from core

- Import-time `pip.main([...])` installation.
- Hard-coded `/content/drive/...` file paths.
- Global notebook state such as `_adata`.
- Class-level shared dataset containers.
- Benchmark/comparison wrappers for Harmony, MNN, Scanorama, SCVI, MARIO, MOFA, Mowgli, totalVI, and SCIB.
- `Matching_backup.py` and inactive matching alternatives.
- Debug-only plotting/output such as unconditional `relations.png`.
- Mandatory label-dependent evaluation in the core pipeline.

## Preserved / corrected

### SCPRO-HI

- AnnData is the internal unit of work.
- The active horizontal integration logic is kept: per-batch clustering, distinctive feature selection, cluster matching, cell anchor construction, and triplet VAE correction.
- `n_features` is no longer overwritten by `adata.n_vars`.
- The corrected matrix starts from original `X`, so unselected features are preserved instead of becoming zero.
- `label_key` is optional and diagnostics-only.

### SCPRO-VI

- MuData is the internal unit of work.
- Combined AnnData can be converted to MuData via flexible modality labels.
- PPI input is explicit (`ppi` or `ppi_path`).
- Cell-cell similarity matrices are stored in `.obsp`; embeddings are stored in `.obsm`.
- Protein similarity uses the original PPI-weighted outer-product distance but computes it in a blockwise memory-aware way.
