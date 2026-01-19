#!/usr/bin/env python3
"""
Data Quality Validator for Training System

Ensures only high-quality, bird-only point cloud data is used for training.
Filters out bad data, validates 3D structure, and checks for anomalies.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """Validates point cloud data for training quality."""

    def __init__(self, config: Dict):
        """Initialize validator with quality thresholds."""
        self.filters = config.get("filters", {})
        self.bird_validation = config.get("bird_validation", {})

    def validate_point_cloud(
        self,
        points: np.ndarray,
        species: str,
        yolo_confidence: float = 0.0,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str, float]:
        """
        Validate a point cloud for training quality.

        Args:
            points: Nx3 numpy array of 3D points
            species: Bird species name
            yolo_confidence: YOLO detection confidence
            metadata: Additional metadata

        Returns:
            (is_valid, reason, quality_score)
        """
        if points is None or len(points) == 0:
            return False, "empty_point_cloud", 0.0

        quality_scores = []

        # 1. Point count check
        num_points = len(points)
        min_pts = self.filters.get("min_points", 200)
        max_pts = self.filters.get("max_points", 10000)

        if num_points < min_pts:
            return False, f"too_few_points ({num_points} < {min_pts})", 0.0
        if num_points > max_pts:
            return False, f"too_many_points ({num_points} > {max_pts})", 0.0

        # Score based on point count (prefer 1000-3000 range)
        if 1000 <= num_points <= 3000:
            quality_scores.append(1.0)
        elif 500 <= num_points < 1000 or 3000 < num_points <= 5000:
            quality_scores.append(0.8)
        else:
            quality_scores.append(0.6)

        # 2. Planarity check (reject flat surfaces)
        planarity = self._compute_planarity(points)
        max_planarity = self.filters.get("planarity_max", 0.3)

        if planarity > max_planarity:
            return False, f"too_planar ({planarity:.3f} > {max_planarity})", 0.0

        quality_scores.append(1.0 - planarity / max_planarity)

        # 3. Thickness check (3D volume)
        thickness = self._compute_thickness(points)
        min_thick = self.filters.get("thickness_min", 0.01)
        max_thick = self.filters.get("thickness_max", 0.30)

        if thickness < min_thick:
            return False, f"too_thin ({thickness:.3f}m < {min_thick}m)", 0.0
        if thickness > max_thick:
            return False, f"too_thick ({thickness:.3f}m > {max_thick}m)", 0.0

        # Score thickness (prefer 0.05-0.15m range for small birds)
        if 0.05 <= thickness <= 0.15:
            quality_scores.append(1.0)
        elif 0.03 <= thickness < 0.05 or 0.15 < thickness <= 0.20:
            quality_scores.append(0.8)
        else:
            quality_scores.append(0.6)

        # 4. Size check (bounding box)
        size = self._compute_size(points)
        min_size = self.filters.get("size_min_m", 0.03)
        max_size = self.filters.get("size_max_m", 0.35)

        if size < min_size:
            return False, f"too_small ({size:.3f}m < {min_size}m)", 0.0
        if size > max_size:
            return False, f"too_large ({size:.3f}m > {max_size}m)", 0.0

        quality_scores.append(0.9)  # Passed size check

        # 5. YOLO confidence check
        if self.bird_validation.get("use_yolo_confidence", True):
            min_conf = self.bird_validation.get("min_yolo_confidence", 0.6)
            if yolo_confidence < min_conf:
                return False, f"low_yolo_confidence ({yolo_confidence:.3f} < {min_conf})", 0.0
            quality_scores.append(min(1.0, yolo_confidence / 0.9))

        # 6. Aspect ratio check (bird-like proportions)
        if self.bird_validation.get("check_aspect_ratio", True):
            aspect_ok, aspect_score = self._check_aspect_ratio(points)
            if not aspect_ok:
                return False, "invalid_aspect_ratio", 0.0
            quality_scores.append(aspect_score)

        # 7. 3D structure check (not just noise)
        if self.bird_validation.get("check_3d_structure", True):
            structure_ok, structure_score = self._check_3d_structure(points)
            if not structure_ok:
                return False, "invalid_3d_structure", 0.0
            quality_scores.append(structure_score)

        # 8. Anomaly detection (outliers, weird shapes)
        if self.bird_validation.get("reject_anomalies", True):
            has_anomaly, anomaly_score = self._detect_anomalies(points)
            if has_anomaly:
                return False, "anomaly_detected", 0.0
            quality_scores.append(anomaly_score)

        # Compute overall quality score
        quality_score = np.mean(quality_scores)

        # Check against minimum threshold
        quality_threshold = self.filters.get("min_quality_score", 0.7)
        if quality_score < quality_threshold:
            return False, f"low_quality_score ({quality_score:.3f} < {quality_threshold})", quality_score

        return True, "valid", quality_score

    def _compute_planarity(self, points: np.ndarray) -> float:
        """Compute planarity using PCA (ratio of smallest to largest eigenvalue)."""
        try:
            centered = points - points.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending

            if eigenvalues[0] < 1e-6:
                return 1.0  # Degenerate case

            planarity = eigenvalues[2] / eigenvalues[0]
            return float(planarity)
        except Exception:
            return 1.0  # Assume planar if calculation fails

    def _compute_thickness(self, points: np.ndarray) -> float:
        """Compute thickness along smallest principal axis."""
        try:
            centered = points - points.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)

            # Project onto smallest eigenvector
            smallest_axis = eigenvectors[:, 0]
            projections = np.dot(centered, smallest_axis)
            thickness = projections.max() - projections.min()

            return float(thickness)
        except Exception:
            return 0.0

    def _compute_size(self, points: np.ndarray) -> float:
        """Compute maximum dimension of bounding box."""
        try:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            dims = maxs - mins
            return float(dims.max())
        except Exception:
            return 0.0

    def _check_aspect_ratio(self, points: np.ndarray) -> Tuple[bool, float]:
        """Check if aspect ratio is bird-like."""
        try:
            mins = points.min(axis=0)
            maxs = points.max(axis=0)
            dims = maxs - mins
            dims = np.sort(dims)[::-1]  # Descending

            if dims[0] < 1e-6:
                return False, 0.0

            # Birds typically have length > width > height
            # Ratios should be reasonable (not too extreme)
            ratio1 = dims[1] / dims[0]  # width/length
            ratio2 = dims[2] / dims[0]  # height/length

            # Expect roughly: 0.5 < width/length < 0.9
            #                 0.4 < height/length < 0.9
            if 0.4 <= ratio1 <= 0.95 and 0.3 <= ratio2 <= 0.95:
                # Score based on how typical the ratios are
                score = 1.0 - abs(ratio1 - 0.7) - abs(ratio2 - 0.6)
                score = max(0.5, min(1.0, score))
                return True, score
            else:
                return False, 0.0

        except Exception:
            return False, 0.0

    def _check_3d_structure(self, points: np.ndarray) -> Tuple[bool, float]:
        """Check if point cloud has proper 3D structure (not just noise)."""
        try:
            # Check density - points should be somewhat clustered
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)

            # Compute coefficient of variation
            mean_dist = distances.mean()
            std_dist = distances.std()

            if mean_dist < 1e-6:
                return False, 0.0

            cv = std_dist / mean_dist

            # Good structure: 0.3 < CV < 0.8 (not too uniform, not too scattered)
            if 0.2 <= cv <= 1.0:
                score = 1.0 - abs(cv - 0.5) / 0.5
                score = max(0.5, min(1.0, score))
                return True, score
            else:
                return False, 0.0

        except Exception:
            return False, 0.0

    def _detect_anomalies(self, points: np.ndarray) -> Tuple[bool, float]:
        """Detect anomalies using statistical outliers."""
        try:
            centroid = points.mean(axis=0)
            distances = np.linalg.norm(points - centroid, axis=1)

            # Use MAD (Median Absolute Deviation)
            median_dist = np.median(distances)
            mad = np.median(np.abs(distances - median_dist))

            if mad < 1e-6:
                # Too uniform - suspicious
                return True, 0.0

            # Count outliers (>3 MAD from median)
            outlier_threshold = median_dist + 3 * 1.4826 * mad
            num_outliers = np.sum(distances > outlier_threshold)
            outlier_ratio = num_outliers / len(points)

            # Reject if >10% outliers
            if outlier_ratio > 0.10:
                return True, 0.0

            # Score based on outlier ratio
            score = 1.0 - outlier_ratio * 5  # Scale 0-10% to 1.0-0.5
            score = max(0.5, min(1.0, score))

            return False, score

        except Exception:
            return True, 0.0


class TrainingDataset:
    """Manages validated training data."""

    def __init__(self, data_dir: Path, validator: DataValidator):
        """Initialize dataset manager."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.validator = validator
        self.index_file = self.data_dir / "index.json"

    def add_sample(
        self,
        points: np.ndarray,
        species: str,
        yolo_confidence: float,
        timestamp: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add a validated sample to the dataset."""
        # Validate first
        is_valid, reason, quality_score = self.validator.validate_point_cloud(
            points, species, yolo_confidence, metadata
        )

        if not is_valid:
            logger.debug(f"Rejected sample for {species}: {reason}")
            return False

        # Save the sample
        try:
            import json
            from datetime import datetime

            # Create species directory
            species_dir = self.data_dir / species.replace(" ", "_")
            species_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename
            timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
            sample_id = f"{timestamp_str}_{int(quality_score*1000):03d}"
            ply_path = species_dir / f"{sample_id}.npy"

            # Save point cloud
            np.save(ply_path, points.astype(np.float32))

            # Update index
            index = self._load_index()
            if species not in index:
                index[species] = []

            index[species].append({
                "sample_id": sample_id,
                "path": str(ply_path.relative_to(self.data_dir)),
                "timestamp": timestamp,
                "quality_score": quality_score,
                "yolo_confidence": yolo_confidence,
                "num_points": len(points),
                "metadata": metadata or {}
            })

            self._save_index(index)

            logger.info(f"Added training sample for {species}: {sample_id} (quality={quality_score:.3f})")
            return True

        except Exception as e:
            logger.error(f"Failed to save sample: {e}")
            return False

    def _load_index(self) -> Dict:
        """Load dataset index."""
        import json
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self, index: Dict):
        """Save dataset index."""
        import json
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def get_species_count(self) -> Dict[str, int]:
        """Get count of samples per species."""
        index = self._load_index()
        return {species: len(samples) for species, samples in index.items()}

    def is_ready_for_training(self, min_samples_per_species: int = 50) -> Tuple[bool, Dict]:
        """Check if dataset has enough samples for training."""
        counts = self.get_species_count()

        if not counts:
            return False, {"reason": "no_data"}

        # Check minimum samples
        ready_species = {sp: cnt for sp, cnt in counts.items() if cnt >= min_samples_per_species}

        if not ready_species:
            return False, {
                "reason": "insufficient_samples",
                "current": counts,
                "required": min_samples_per_species
            }

        return True, {
            "ready_species": ready_species,
            "total_samples": sum(counts.values())
        }
