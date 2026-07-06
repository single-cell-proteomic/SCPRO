import scanpy as sc
import scpro as sp

adata = sc.read_h5ad("data/cite_seq_combined.h5ad")
mdata = sp.vi.from_combined_anndata(
    adata,
    modality_key="feature_type",
    rna_values=("rna", "Gene Expression"),
    protein_values=("protein", "ADT", "Antibody Capture"),
)

sp.vi.run(mdata, ppi_path="ppi_weights.csv", result_key="X_scpro_vi", random_state=42)
print(mdata.obsm["X_scpro_vi"].shape)
