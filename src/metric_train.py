import argparse
import os
from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from pytorch_metric_learning import losses

from src.pc_preprocess import preprocess, DEFAULT_POINTS
from src.models.dgcnn import DGCNN


class PointCloudDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        root_path = Path(root)
        for class_dir in root_path.iterdir():
            if not class_dir.is_dir():
                continue
            for indiv_dir in class_dir.iterdir():
                if not indiv_dir.is_dir():
                    continue
                ply_dir = indiv_dir / "samples"
                for ply in ply_dir.glob("*.ply"):
                    self.samples.append((ply, class_dir.name, indiv_dir.name))
        if not self.samples:
            raise ValueError("No samples found; expected data/<class>/<individual>/samples/*.ply")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_name, indiv_id = self.samples[idx]
        pc = np.asarray(o3d.io.read_point_cloud(str(path)).points)
        pc = preprocess(pc, num_points=DEFAULT_POINTS)
        return torch.from_numpy(pc).float(), class_name, indiv_id


class MetricTrainer:
    def __init__(self, data_root: str, batch_size: int = 4, lr: float = 1e-3, emb_dims: int = 256,
                 loss_name: str = "triplet"):
        self.dataset = PointCloudDataset(data_root)
        self.loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DGCNN(emb_dims=emb_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        if loss_name == "supcon":
            self.criterion = losses.SupConLoss()
        else:
            self.criterion = losses.TripletMarginLoss(margin=0.2)

    def train_epoch(self):
        self.model.train()
        total = 0.0
        for pcs, class_names, indiv_ids in self.loader:
            pcs = pcs.to(self.device)
            embeds = self.model(pcs)
            # Labels combine class + indiv to keep positive pairs within identity
            labels = [f"{c}:{i}" for c, i in zip(class_names, indiv_ids)]
            labels = torch.tensor([hash(l) % (2 ** 16) for l in labels], device=self.device)
            loss = self.criterion(embeds, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        return total / len(self.loader)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)


def parse_args():
    parser = argparse.ArgumentParser(description="Metric learning trainer for point-cloud re-id")
    parser.add_argument("--data", required=True, help="Dataset root: data/<class>/<individual>/samples/*.ply")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb-dims", type=int, default=256)
    parser.add_argument("--loss", choices=["triplet", "supcon"], default="triplet")
    parser.add_argument("--output", default="checkpoints/reid.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trainer = MetricTrainer(args.data, batch_size=args.batch_size, lr=args.lr, emb_dims=args.emb_dims, loss_name=args.loss)
    for epoch in range(args.epochs):
        loss = trainer.train_epoch()
        print(f"Epoch {epoch+1}/{args.epochs} loss={loss:.4f}")
    trainer.save(args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    import numpy as np
    main()

