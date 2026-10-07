import torch
import torch.nn as nn


class PathEncoder(nn.Module):
    """One TokenLearner path: patch embedding + learnable token + Transformer encoder."""
    def __init__(self, in_dim, n_patches, D=600, L=6, H=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, D)
        self.token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, D))
        nn.init.trunc_normal_(self.token, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=H, dim_feedforward=4 * D, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=L, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D)

    def forward(self, x):                                   # x: (B, n_patches, in_dim)
        x = self.embed(x)
        tok = self.token.expand(x.size(0), -1, -1)
        x = torch.cat([tok, x], dim=1) + self.pos
        x = self.encoder(x)
        return self.norm(x[:, 0])                           # learned token: (B, D)


class DualTL(nn.Module):
    def __init__(self, N=63, C=3, T=300, D=600, L=6, H=4, dropout=0.1):
        super().__init__()
        self.spatial = PathEncoder(C * T, N, D, L, H, dropout)    # 63 patches of 900 values
        self.temporal = PathEncoder(N * C, T, D, L, H, dropout)   # 300 patches of 189 values
        self.head = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, T))

    def forward(self, m):                                   # m: (B, N, C, T)
        xs = m.flatten(2)                                   # (B, N, C*T)
        xt = m.permute(0, 3, 1, 2).flatten(2)               # (B, T, N*C)
        t_dual = torch.cat([self.spatial(xs), self.temporal(xt)], dim=1)   # (B, 2D)
        return self.head(t_dual)                            # (B, T)


def neg_pearson(pred, gt):
    """Negative Pearson correlation loss, averaged over the batch."""
    p = pred - pred.mean(dim=1, keepdim=True)
    g = gt - gt.mean(dim=1, keepdim=True)
    r = (p * g).sum(1) / (p.norm(dim=1) * g.norm(dim=1) + 1e-8)
    return (1 - r).mean()