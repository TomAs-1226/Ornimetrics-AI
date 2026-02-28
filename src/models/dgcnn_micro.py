"""Ultra-light DGCNN variant for Raspberry Pi.

~35K parameters (vs 137K light, 3.5M heavy).
Designed for real-time inference on ARM CPUs without GPU.
Uses k=10 neighbors, 64-dim embeddings, and a single graph conv block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNNMicro(nn.Module):
    def __init__(self, emb_dims: int = 64, k: int = 10, input_dims: int = 3):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.input_dims = input_dims
        edge_channels = 2 * input_dims

        self.conv1 = nn.Sequential(
            nn.Conv2d(edge_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def knn(self, x):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        k = min(self.k, max(1, x.size(2) - 1))
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx, k

    def get_graph_feature(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        idx, k = self.knn(x)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, -1)
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.get_graph_feature(x)
        x = self.conv1(x)
        x = self.conv2(x).max(dim=-1, keepdim=False)[0]
        x = self.conv3(x)
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x


__all__ = ["DGCNNMicro"]
