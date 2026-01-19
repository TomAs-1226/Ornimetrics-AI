#!/usr/bin/env python3
"""
3D Point Cloud Model Trainer

Trains DGCNN models on collected bird data for improved individual recognition.
Runs during night-time hours on CPU while Hailo handles daytime detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class BirdPointCloudDataset(Dataset):
    """PyTorch dataset for bird point clouds."""

    def __init__(self, data_dir: Path, species_list: List[str], augment: bool = False):
        """Initialize dataset from validated training data."""
        self.data_dir = Path(data_dir)
        self.species_list = species_list
        self.species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
        self.augment = augment
        self.samples = []

        # Load index
        index_file = self.data_dir / "index.json"
        if not index_file.exists():
            raise ValueError(f"No index found at {index_file}")

        with open(index_file, 'r') as f:
            index = json.load(f)

        # Collect samples
        for species in species_list:
            if species not in index:
                logger.warning(f"No samples found for species: {species}")
                continue

            for sample_info in index[species]:
                sample_path = self.data_dir / sample_info["path"]
                if sample_path.exists():
                    self.samples.append({
                        "path": sample_path,
                        "species": species,
                        "label": self.species_to_idx[species],
                        "quality": sample_info.get("quality_score", 1.0)
                    })

        logger.info(f"Loaded {len(self.samples)} samples across {len(species_list)} species")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load point cloud
        points = np.load(sample["path"])

        # Augmentation
        if self.augment:
            points = self._augment(points)

        # Normalize
        points = self._normalize(points)

        # Convert to tensor
        points = torch.from_numpy(points).float()
        label = torch.tensor(sample["label"], dtype=torch.long)

        return points, label

    def _normalize(self, points: np.ndarray) -> np.ndarray:
        """Center and optionally scale point cloud."""
        # Center
        centroid = points.mean(axis=0)
        points = points - centroid

        return points

    def _augment(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation."""
        # Random rotation around Z axis (up)
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        points = points @ R.T

        # Random jitter
        jitter = np.random.normal(0, 0.005, points.shape)
        points = points + jitter

        # Random scale
        scale = np.random.uniform(0.95, 1.05)
        points = points * scale

        return points


class ModelTrainer:
    """Handles model training, validation, and checkpointing."""

    def __init__(self, config: Dict, device: str = "cpu"):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device(device)

        # Training parameters
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 16)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.early_stopping_patience = config.get("early_stopping_patience", 10)
        self.checkpoint_every = config.get("checkpoint_every_n_epochs", 5)

        # Paths
        self.checkpoint_dir = Path("models/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.model = None
        self.optimizer = None
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epochs": []
        }

    def prepare_data(
        self,
        data_dir: Path,
        species_list: List[str],
        val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation dataloaders."""
        # Create full dataset
        full_dataset = BirdPointCloudDataset(data_dir, species_list, augment=True)

        # Split into train and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=False  # CPU training
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )

        logger.info(f"Training samples: {train_size}, Validation samples: {val_size}")

        return train_loader, val_loader

    def initialize_model(self, num_classes: int, embedding_dim: int = 512):
        """Initialize DGCNN model."""
        # Import model (assuming it exists in src/models/)
        try:
            from src.models.dgcnn import DGCNNHeavy

            self.model = DGCNNHeavy(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                k=40
            )
            self.model = self.model.to(self.device)

            logger.info(f"Initialized DGCNNHeavy with {num_classes} classes, {embedding_dim}D embeddings")

        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        for batch_idx, (points, labels) in enumerate(train_loader):
            points = points.to(self.device)
            labels = labels.to(self.device)

            # Transpose points to (B, 3, N) format expected by DGCNN
            points = points.transpose(1, 2)

            # Forward pass
            self.optimizer.zero_grad()
            logits, embeddings = self.model(points)

            # Compute loss
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.device)
                labels = labels.to(self.device)

                # Transpose
                points = points.transpose(1, 2)

                # Forward pass
                logits, embeddings = self.model(points)

                # Compute loss
                loss = criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> Dict:
        """Run full training loop."""
        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Track history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["train_acc"].append(train_acc)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_acc"].append(val_acc)
            self.training_history["epochs"].append(epoch)

            epoch_time = time.time() - epoch_start

            logger.info(
                f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s): "
                f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%"
            )

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
                logger.info(f"New best model! Val loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1

            # Regular checkpoint
            if epoch % self.checkpoint_every == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")

        return {
            "final_epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_acc": max(self.training_history["val_acc"]),
            "training_time_hours": total_time / 3600,
            "history": self.training_history
        }

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            # Save as best model
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

        return checkpoint["epoch"]
