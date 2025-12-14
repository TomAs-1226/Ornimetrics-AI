import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset
import open3d as o3d
from pytorch_metric_learning import losses, miners

from src.pc_preprocess import preprocess_with_stats, DEFAULT_POINTS, PreprocessConfig
from src.models.dgcnn import DGCNN


def load_ply(path: Path) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(path))
    return np.asarray(pc.points)


@dataclass
class Sample:
    path: Path
    species: str
    individual: str


class SpeciesBatchDataset(Dataset):
    def __init__(self, root: str, preprocess_cfg: PreprocessConfig):
        self.root = Path(root)
        self.preprocess_cfg = preprocess_cfg
        self.by_species: Dict[str, List[Sample]] = {}
        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue
            species = class_dir.name
            for indiv_dir in class_dir.iterdir():
                if not indiv_dir.is_dir():
                    continue
                indiv_id = indiv_dir.name
                for ply in (indiv_dir / "samples").glob("*.ply"):
                    self.by_species.setdefault(species, []).append(Sample(ply, species, indiv_id))
        if not self.by_species:
            raise ValueError("No samples found; expected data/<class>/<individual>/samples/*.ply")
        self.species_list = list(self.by_species.keys())

    def sample_batch(self, species: str, k_individuals: int, m_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng()
        samples = self.by_species[species]
        indivs = list({s.individual for s in samples})
        rng.shuffle(indivs)
        indivs = indivs[:k_individuals]
        pcs = []
        labels = []
        for indiv in indivs:
            indiv_samples = [s for s in samples if s.individual == indiv]
            rng.shuffle(indiv_samples)
            indiv_samples = indiv_samples[:m_samples]
            for s in indiv_samples:
                pc = load_ply(s.path)
                prep = preprocess_with_stats(pc, self.preprocess_cfg)
                pcs.append(prep.points)
                labels.append(indiv)
        pcs = np.stack(pcs)
        pcs = torch.from_numpy(pcs).float()
        label_tensor = torch.tensor([hash(f"{species}:{l}") % (2 ** 16) for l in labels], dtype=torch.long)
        return pcs, label_tensor

    def __len__(self):
        return 1000000  # sampled dynamically

    def __getitem__(self, idx):  # unused
        raise NotImplementedError


@dataclass
class TrainConfig:
    k_individuals: int = 8
    m_samples: int = 4
    loss: str = "supcon"
    lr: float = 1e-3
    epochs: int = 5


class MetricTrainer:
    def __init__(self, data_root: str, batch_size: int = 4, emb_dims: int = 256, config: TrainConfig = None, preprocess_cfg: PreprocessConfig = None):
        self.config = config or TrainConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.preprocess_cfg = preprocess_cfg or PreprocessConfig(fps_points=DEFAULT_POINTS)
        self.dataset = SpeciesBatchDataset(data_root, self.preprocess_cfg)
        self.batch_size = batch_size
        self.model = DGCNN(emb_dims=emb_dims).to(self.device)
        if self.config.loss == "supcon":
            self.criterion = losses.SupConLoss()
        elif self.config.loss == "arcface":
            self.criterion = losses.ArcFaceLoss(num_classes=10000, embedding_size=emb_dims)
        else:
            self.criterion = losses.TripletMarginLoss(margin=0.2)
            self.miner = miners.TripletMarginMiner(margin=0.2, type_of_triplets="semihard")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

    def _sample_batch(self):
        pcs = []
        labels = []
        for _ in range(self.batch_size):
            species = np.random.choice(self.dataset.species_list)
            pc, lab = self.dataset.sample_batch(species, self.config.k_individuals, self.config.m_samples)
            pcs.append(pc)
            labels.append(lab)
        pcs = torch.cat(pcs, dim=0)
        labels = torch.cat(labels, dim=0)
        return pcs, labels

    def train_epoch(self):
        self.model.train()
        total = 0.0
        steps = 0
        for _ in range(self.config.k_individuals):
            pcs, labels = self._sample_batch()
            pcs, labels = pcs.to(self.device), labels.to(self.device)
            embeds = self.model(pcs)
            if self.config.loss == "triplet":
                hard_pairs = self.miner(embeds, labels)
                loss = self.criterion(embeds, labels, hard_pairs)
            else:
                loss = self.criterion(embeds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
            steps += 1
        return total / max(1, steps)

    @torch.no_grad()
    def eval_recall(self, val_pairs: List[Tuple[torch.Tensor, torch.Tensor]]):
        if not val_pairs:
            return {"Recall@1": 0.0, "Recall@5": 0.0, "mAP": 0.0}
        self.model.eval()
        recalls = []
        for pcs, labels in val_pairs:
            pcs, labels = pcs.to(self.device), labels.to(self.device)
            embeds = self.model(pcs)
            sim = torch.matmul(embeds, embeds.T)
            preds = torch.argsort(sim, dim=1, descending=True)
            hits1 = (labels[preds[:, 1]] == labels).float().mean().item()
            hits5 = []
            for i in range(labels.size(0)):
                top5 = preds[i, 1:6]
                hits5.append((labels[top5] == labels[i]).any().float().item())
            mAP = hits5.count(1.0) / max(1, len(hits5))
            recalls.append({"Recall@1": hits1, "Recall@5": float(np.mean(hits5)), "mAP": mAP})
        avg = {k: float(np.mean([r[k] for r in recalls])) for k in recalls[0]}
        return avg

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)


def parse_args():
    parser = argparse.ArgumentParser(description="Metric learning trainer for point-cloud re-id")
    parser.add_argument("--data", required=True, help="Dataset root: data/<class>/<individual>/samples/*.ply")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dims", type=int, default=256)
    parser.add_argument("--loss", choices=["triplet", "supcon", "arcface"], default="supcon")
    parser.add_argument("--output", default="checkpoints/reid.pt")
    parser.add_argument("--k-individuals", type=int, default=8)
    parser.add_argument("--m-samples", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig(k_individuals=args.k_individuals, m_samples=args.m_samples, loss=args.loss, lr=args.lr, epochs=args.epochs)
    preprocess_cfg = PreprocessConfig(fps_points=DEFAULT_POINTS)
    trainer = MetricTrainer(args.data, batch_size=args.batch_size, emb_dims=args.emb_dims, config=cfg, preprocess_cfg=preprocess_cfg)
    metrics_log = []
    for epoch in range(args.epochs):
        loss = trainer.train_epoch()
        metrics = trainer.eval_recall([])
        metrics_log.append({"epoch": epoch + 1, "loss": loss, **metrics})
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f} recall1={metrics['Recall@1']:.3f} recall5={metrics['Recall@5']:.3f}")
    trainer.save(args.output)
    os.makedirs("runs/train_reid", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(Path("runs/train_reid") / f"{ts}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
