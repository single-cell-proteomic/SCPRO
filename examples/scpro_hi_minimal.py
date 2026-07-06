import scpro as sp

adata = sp.hi.run(
    "data/protein_batches/",
    batch_key="batch_id",
    label_key="cell_type",
    n_features=38,
    result_key="X_scpro_hi",
    random_state=42,
)
print(adata.obsm["X_scpro_hi"].shape)
