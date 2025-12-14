from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch

from src.pc_preprocess import preprocess_with_stats, DEFAULT_POINTS, PreprocessConfig
from src.models.dgcnn import DGCNN


@dataclass
class EmbedResult:
    embedding: torch.Tensor
    preprocess_stats: Dict[str, Any]


class PointReID:
    def __init__(self, model_path: Optional[str] = None, device: str = None, emb_dims: int = 256, preprocess_config: Optional[PreprocessConfig] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DGCNN(emb_dims=emb_dims).to(self.device)
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()
        self.preprocess_config = preprocess_config or PreprocessConfig(fps_points=DEFAULT_POINTS)

    @torch.no_grad()
    def embed(self, point_cloud) -> EmbedResult:
        prep = preprocess_with_stats(point_cloud, self.preprocess_config)
        if prep.points.size == 0:
            embedding = torch.zeros(self.model.emb_dims, device=self.device)
        else:
            tensor = torch.from_numpy(prep.points).float().unsqueeze(0).to(self.device)
            embedding = self.model(tensor).squeeze(0)
        norm = torch.norm(embedding) + 1e-8
        return EmbedResult(embedding=embedding / norm, preprocess_stats=prep.stats)
