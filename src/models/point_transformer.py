import torch
import torch.nn as nn
import torch.nn.functional as F


class PointTransformerBlock(nn.Module):
    def __init__(self, dim: int, k: int = 16):
        super().__init__()
        self.k = k
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        # pairwise distances
        with torch.no_grad():
            dist = torch.cdist(x, x)
            idx = dist.topk(k=min(self.k, max(1, n - 1)), largest=False).indices  # (b, n, k)
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        gathered_k = torch.gather(k.unsqueeze(1).expand(-1, n, -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, idx.size(-1), c))
        gathered_v = torch.gather(v.unsqueeze(1).expand(-1, n, -1, -1), 2, idx.unsqueeze(-1).expand(-1, -1, idx.size(-1), c))
        attn = torch.softmax((q.unsqueeze(2) * gathered_k).sum(-1, keepdim=True) / (c ** 0.5), dim=2)
        agg = (attn * gathered_v).sum(2)
        out = x + agg
        return out + self.ffn(out)


class PointTransformerLarge(nn.Module):
    """Larger-capacity transformer style encoder for point re-id."""

    def __init__(self, emb_dims: int = 512, input_dims: int = 3, width: int = 128, depth: int = 4, k: int = 16):
        super().__init__()
        self.emb_dims = emb_dims
        self.input_proj = nn.Linear(input_dims, width)
        self.blocks = nn.ModuleList([PointTransformerBlock(width, k=k) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.ReLU(inplace=True),
            nn.Linear(width * 2, emb_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.max(dim=1)[0]
        x = self.head(x)
        x = F.normalize(x, p=2, dim=1)
        return x


__all__ = ["PointTransformerLarge"]
