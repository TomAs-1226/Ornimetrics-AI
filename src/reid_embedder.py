from typing import Optional
import torch
from src.pc_preprocess import preprocess, DEFAULT_POINTS
from src.models.dgcnn import DGCNN


class PointReID:
    def __init__(self, model_path: Optional[str] = None, device: str = None, emb_dims: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DGCNN(emb_dims=emb_dims).to(self.device)
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def embed(self, point_cloud) -> torch.Tensor:
        pc = preprocess(point_cloud, num_points=DEFAULT_POINTS)
        if pc.size == 0:
            return torch.zeros(self.model.conv5[0].out_channels, device=self.device)
        tensor = torch.from_numpy(pc).float().unsqueeze(0).to(self.device)
        embedding = self.model(tensor)
        return embedding.squeeze(0)

