from __future__ import annotations


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.nn import SAGEConv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SCPRO-VI training requires PyTorch and torch-geometric. Install scpro[vi]."
        ) from exc
    return torch, nn, SAGEConv


def kl_loss(mu, logvar, n_nodes: int):
    torch, _, _ = _torch_modules()
    logvar = logvar.clamp(max=10)
    return 1 / n_nodes * (-0.5 * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), dim=1)))


def recon_loss(preds, adj):
    torch, _, _ = _torch_modules()
    neg_mask = adj < 0.95
    diff = torch.abs(preds - adj)
    return diff[neg_mask].mean() + diff[~neg_mask].mean()


def build_sgvae_class():
    torch, nn, SAGEConv = _torch_modules()

    class SGVAE(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
            super().__init__()
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, latent_dim)
            self.conv3 = SAGEConv(hidden_dim, latent_dim)

        def set_optimizer(self, params):
            return torch.optim.Adam(params)

        def encode(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            mu = self.conv2(x, edge_index)
            logvar = self.conv3(x, edge_index)
            return mu, logvar

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            eps = 1e-8
            norm_z = z / (torch.norm(z, dim=1, keepdim=True) + eps)
            return torch.mm(norm_z, norm_z.t())

        def forward(self, x, edge_index):
            mu, logvar = self.encode(x, edge_index)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar, z

    return SGVAE


def build_graphvae_class():
    torch, nn, SAGEConv = _torch_modules()

    class GraphVAE(nn.Module):
        def __init__(self, prot_dim, rna_dim, hidden_dim, latent_dim, pretrained=False, p_vgae=None, r_vgae=None):
            super().__init__()
            if pretrained:
                if p_vgae is None or r_vgae is None:
                    raise ValueError("pretrained=True requires pretrained protein and RNA SGVAE models.")
                self.conv1_p = p_vgae.conv1
                self.conv2_p = p_vgae.conv2
                self.conv3_p = p_vgae.conv3
                self.conv1_r = r_vgae.conv1
                self.conv2_r = r_vgae.conv2
                self.conv3_r = r_vgae.conv3
            else:
                self.conv1_p = SAGEConv(prot_dim, hidden_dim)
                self.conv1_r = SAGEConv(rna_dim, hidden_dim)
                self.conv2_p = SAGEConv(hidden_dim, latent_dim)
                self.conv2_r = SAGEConv(hidden_dim, latent_dim)
                self.conv3_p = SAGEConv(hidden_dim, latent_dim)
                self.conv3_r = SAGEConv(hidden_dim, latent_dim)
            self.fc_decode = nn.Linear(2 * latent_dim, latent_dim)
            self.fc_decode_last = nn.Linear(latent_dim, int(latent_dim / 2))

        def set_optimizer(self, params):
            return torch.optim.Adam(params)

        def encode(self, x_p, x_r, edge_index_p, edge_index_r):
            x_p = self.conv1_p(x_p, edge_index_p).relu()
            mu_p = self.conv2_p(x_p, edge_index_p)
            logvar_p = self.conv3_p(x_p, edge_index_p)
            x_r = self.conv1_r(x_r, edge_index_r).relu()
            mu_r = self.conv2_r(x_r, edge_index_r)
            logvar_r = self.conv3_r(x_r, edge_index_r)
            return mu_p, mu_r, logvar_p, logvar_r

        def reparameterize(self, mu_p, mu_r, logvar_p, logvar_r):
            z_p = mu_p + torch.randn_like(logvar_p) * torch.exp(0.5 * logvar_p)
            z_r = mu_r + torch.randn_like(logvar_r) * torch.exp(0.5 * logvar_r)
            return z_p, z_r

        def concat_z(self, z_p, z_r):
            z = torch.cat((z_p, z_r), dim=1)
            z = self.fc_decode(z)
            return self.fc_decode_last(z)

        def decode(self, z):
            eps = 1e-8
            norm_z = z / (torch.norm(z, dim=1, keepdim=True) + eps)
            return torch.mm(norm_z, norm_z.t())

        def forward(self, x_p, x_r, edge_index_p, edge_index_r):
            mu_p, mu_r, logvar_p, logvar_r = self.encode(x_p, x_r, edge_index_p, edge_index_r)
            z_p, z_r = self.reparameterize(mu_p, mu_r, logvar_p, logvar_r)
            z = self.concat_z(z_p, z_r)
            return self.decode(z_p), self.decode(z_r), self.decode(z), mu_p, mu_r, logvar_p, logvar_r, z

    return GraphVAE
