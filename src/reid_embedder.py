from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import torch

from src.pc_preprocess import preprocess_with_stats, DEFAULT_POINTS, PreprocessConfig
from src.models.dgcnn import DGCNN
from src.models.dgcnn_heavy import DGCNNHeavy
from src.models.point_transformer import PointTransformerLarge


@dataclass
class EmbedResult:
    embedding: torch.Tensor
    preprocess_stats: Dict[str, Any]
    points: np.ndarray


class PointReID:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = None,
        emb_dims: int = 256,
        preprocess_config: Optional[PreprocessConfig] = None,
        model_name: str = "heavy",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocess_config = preprocess_config or PreprocessConfig(fps_points=DEFAULT_POINTS)
        input_dims = 3 + int(self.preprocess_config.append_scale)
        if model_name == "heavy":
            self.model = DGCNNHeavy(emb_dims=max(emb_dims, 384), input_dims=input_dims).to(self.device)
        elif model_name == "xheavy":
            # Larger point-transformer encoder for offline training/evaluation.
            self.model = PointTransformerLarge(emb_dims=max(emb_dims, 512), input_dims=input_dims).to(self.device)
        else:
            self.model = DGCNN(emb_dims=emb_dims, input_dims=input_dims).to(self.device)
        if model_path:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
        self.model.eval()
        self.expected_input_dims = input_dims

    @torch.no_grad()
    def embed(self, point_cloud) -> EmbedResult:
        prep = preprocess_with_stats(point_cloud, self.preprocess_config)
        points = prep.points
        if points.shape[1] != self.expected_input_dims:
            if points.shape[1] > self.expected_input_dims:
                points = points[:, : self.expected_input_dims]
            else:
                pad = self.expected_input_dims - points.shape[1]
                points = np.pad(points, ((0, 0), (0, pad)), mode="constant")
        if prep.points.size == 0:
            embedding = torch.zeros(self.model.emb_dims, device=self.device)
        else:
            tensor = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
            embedding = self.model(tensor).squeeze(0)
        norm = torch.norm(embedding) + 1e-8
        return EmbedResult(embedding=embedding / norm, preprocess_stats=prep.stats, points=prep.points)
