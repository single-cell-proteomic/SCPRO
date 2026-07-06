from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from scpro._utils import as_dense

from ._graphs import build_graphs
from ._models import build_graphvae_class, build_sgvae_class, kl_loss, recon_loss
from ._schema import validate_mudata


@dataclass
class TrainingConfig:
    latent_dim: int = 100
    hidden_dim: int = 256
    num_neighbors: list[int] | tuple[int, ...] = (15,)
    num_epochs: int = 20
    num_hvgs: int = 1000
    pretrained: bool = True
    use_embeddings: bool = True
    device: str | None = None


def _safe_pca(x: np.ndarray, n_components: int = 25) -> np.ndarray:
    n = min(n_components, max(1, x.shape[0] - 1), max(1, x.shape[1] - 1))
    return PCA(n_components=n, random_state=0).fit_transform(x)


def _rna_matrix(rna, *, use_embeddings: bool, num_hvgs: int) -> np.ndarray:
    if use_embeddings:
        if "embeddings" not in rna.obsm:
            rna.obsm["embeddings"] = _safe_pca(as_dense(rna.X), 25)
        return np.asarray(rna.obsm["embeddings"], dtype=np.float32)
    x = as_dense(rna.X)
    if num_hvgs != -1 and x.shape[1] > num_hvgs:
        # Lightweight variance-based fallback instead of requiring scikit-misc at runtime.
        var = np.var(x, axis=0)
        keep = np.argsort(var)[::-1][:num_hvgs]
        return x[:, keep].astype(np.float32)
    return x.astype(np.float32)


def _edge_tensor(edges, device):
    import torch

    edge_index = torch.as_tensor(edges, dtype=torch.long)
    if edge_index.ndim == 1:
        edge_index = edge_index.reshape(2, -1)
    return edge_index.to(device)


def _pretrain_sgvae(x, edge_index, adj, *, hidden_dim, latent_dim, epochs, device):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    SGVAE = build_sgvae_class()
    model = SGVAE(x.shape[1], hidden_dim, latent_dim).to(device)
    opt = model.set_optimizer(model.parameters())
    data = Data(x=torch.as_tensor(x, dtype=torch.float32), edge_index=edge_index.cpu()).to(device)
    loader = NeighborLoader(data, num_neighbors=[15], batch_size=x.shape[0], shuffle=False)
    z = None
    for _ in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            recon, mu, logvar, z = model(batch.x, batch.edge_index)
            loss = recon_loss(recon, adj) + kl_loss(mu, logvar, x.shape[0])
            loss.backward()
            opt.step()
    return model, z


def train_vi(
    mdata,
    *,
    ppi=None,
    ppi_path=None,
    rna_mod: str = "rna",
    protein_mod: str = "protein",
    result_key: str = "X_scpro_vi",
    graph_key: str = "scpro_vi",
    config: TrainingConfig | None = None,
    ppi_backend: str = "auto",
    protein_block_size: int = 256,
):
    """Train SCPRO-VI and write embeddings into `mdata.obsm[result_key]`."""
    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    validate_mudata(mdata, rna_mod=rna_mod, protein_mod=protein_mod)
    cfg = config or TrainingConfig()
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rna = mdata.mod[rna_mod]
    protein = mdata.mod[protein_mod]
    x_rna = _rna_matrix(rna, use_embeddings=cfg.use_embeddings, num_hvgs=cfg.num_hvgs)
    x_protein = as_dense(protein.X).astype(np.float32)

    if f"{graph_key}_similarity" not in mdata.obsp:
        build_graphs(
            mdata,
            ppi=ppi,
            ppi_path=ppi_path,
            rna_mod=rna_mod,
            protein_mod=protein_mod,
            ppi_backend=ppi_backend,
            protein_block_size=protein_block_size,
            key=graph_key,
        )

    edge_p = _edge_tensor(mdata.uns[graph_key]["protein_edges"], device)
    edge_r = _edge_tensor(mdata.uns[graph_key]["rna_edges"], device)
    adj_similarity = torch.as_tensor(1 - np.asarray(mdata.obsp[f"{graph_key}_similarity"]), dtype=torch.float32, device=device)
    adj_similarity.fill_diagonal_(1)

    prot_adj = torch.as_tensor(np.asarray(protein.obsp[f"{graph_key}_adjacency"]), dtype=torch.float32, device=device)
    rna_adj = torch.as_tensor(np.asarray(rna.obsp[f"{graph_key}_adjacency"]), dtype=torch.float32, device=device)
    prot_adj.fill_diagonal_(1)
    rna_adj.fill_diagonal_(1)

    model_p = model_r = None
    if cfg.pretrained:
        model_p, _ = _pretrain_sgvae(
            x_protein,
            edge_p,
            prot_adj,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            epochs=cfg.num_epochs,
            device=device,
        )
        model_r, _ = _pretrain_sgvae(
            x_rna,
            edge_r,
            rna_adj,
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            epochs=cfg.num_epochs,
            device=device,
        )

    GraphVAE = build_graphvae_class()
    model = GraphVAE(
        x_protein.shape[1],
        x_rna.shape[1],
        cfg.hidden_dim,
        cfg.latent_dim,
        cfg.pretrained,
        model_p,
        model_r,
    ).to(device)
    opt = model.set_optimizer(model.parameters())
    data_p = Data(x=torch.as_tensor(x_protein, dtype=torch.float32), edge_index=edge_p.cpu()).to(device)
    data_r = Data(x=torch.as_tensor(x_rna, dtype=torch.float32), edge_index=edge_r.cpu()).to(device)
    loader_p = NeighborLoader(data_p, num_neighbors=list(cfg.num_neighbors), batch_size=x_protein.shape[0], shuffle=False)
    loader_r = NeighborLoader(data_r, num_neighbors=list(cfg.num_neighbors), batch_size=x_rna.shape[0], shuffle=False)

    z = None
    loss_history = []
    for _ in range(cfg.num_epochs * 2):
        model.train()
        opt.zero_grad()
        for batch_p in loader_p:
            for batch_r in loader_r:
                _, _, recon_joint, mu_p, mu_r, logvar_p, logvar_r, z = model(
                    batch_p.x,
                    batch_r.x,
                    batch_p.edge_index,
                    batch_r.edge_index,
                )
                loss = recon_loss(recon_joint, adj_similarity)
                loss.backward()
                opt.step()
                loss_history.append(float(loss.detach().cpu()))

    if z is None:
        raise RuntimeError("SCPRO-VI training did not produce embeddings.")
    mdata.obsm[result_key] = z.detach().cpu().numpy()
    mdata.uns.setdefault(graph_key, {})
    mdata.uns[graph_key]["result_key"] = result_key
    mdata.uns[graph_key]["loss_history"] = loss_history
    return mdata
