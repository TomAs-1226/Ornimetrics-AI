import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNN(nn.Module):
    def __init__(self, emb_dims: int = 256, k: int = 20):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False), self.bn1, nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), self.bn2, nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False), self.bn3, nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False), self.bn4, nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False), self.bn5, nn.LeakyReLU(negative_slope=0.2))

    def get_graph_feature(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)
        idx = self.knn(x)
        device = x.device
        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)

        x = x.transpose(2, 1).contiguous()
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, self.k, -1)
        x = x.view(batch_size, num_points, 1, -1).repeat(1, 1, self.k, 1)
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
        return feature

    def knn(self, x):
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=self.k, dim=-1)[1]
        return idx

    def forward(self, x):
        x = x.transpose(2, 1)
        x = self.get_graph_feature(x)
        x = self.conv1(x)
        x1 = self.conv2(x).max(dim=-1, keepdim=False)[0]

        x = self.get_graph_feature(x1)
        x = self.conv3(x)
        x2 = self.conv4(x).max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2), dim=1)
        x = self.conv5(x)
        x = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)
        x = F.normalize(x, p=2, dim=1)
        return x


__all__ = ["DGCNN"]

