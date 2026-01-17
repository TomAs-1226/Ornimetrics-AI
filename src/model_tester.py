#!/usr/bin/env python3
"""
Model Testing and Validation System

Performs self-tests on trained models before deployment.
Validates accuracy against baseline data and ensures model quality.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTester:
    """Tests trained models for deployment readiness."""

    def __init__(self, config: Dict, device: str = "cpu"):
        """Initialize tester with configuration."""
        self.config = config
        self.device = torch.device(device)

        # Test parameters
        self.baseline_accuracy_threshold = config.get("baseline_accuracy_threshold", 0.85)
        self.improvement_threshold = config.get("improvement_threshold", 0.02)
        self.test_samples_per_species = config.get("test_samples_per_species", 20)
        self.max_test_time_minutes = config.get("max_test_time_minutes", 30)

        # Test results storage
        self.test_results_dir = Path("models/test_results")
        self.test_results_dir.mkdir(parents=True, exist_ok=True)

    def create_test_set(
        self,
        data_dir: Path,
        species_list: List[str]
    ) -> List[Dict]:
        """Create holdout test set from training data."""
        import random

        test_samples = []

        # Load index
        index_file = data_dir / "index.json"
        with open(index_file, 'r') as f:
            index = json.load(f)

        # Sample test data per species
        for species in species_list:
            if species not in index:
                logger.warning(f"No samples for species {species} in test set")
                continue

            species_samples = index[species]

            # Take top quality samples for testing
            species_samples = sorted(
                species_samples,
                key=lambda x: x.get("quality_score", 0.0),
                reverse=True
            )

            # Take up to test_samples_per_species
            num_test = min(len(species_samples), self.test_samples_per_species)
            test_species_samples = species_samples[:num_test]

            for sample_info in test_species_samples:
                sample_path = data_dir / sample_info["path"]
                if sample_path.exists():
                    test_samples.append({
                        "path": sample_path,
                        "species": species,
                        "quality": sample_info.get("quality_score", 1.0)
                    })

        logger.info(f"Created test set with {len(test_samples)} samples")
        return test_samples

    def test_model(
        self,
        model: torch.nn.Module,
        test_samples: List[Dict],
        species_to_idx: Dict[str, int]
    ) -> Dict:
        """Test model on test set."""
        model.eval()
        model = model.to(self.device)

        correct = 0
        total = 0
        per_species_correct = {sp: 0 for sp in species_to_idx.keys()}
        per_species_total = {sp: 0 for sp in species_to_idx.keys()}

        predictions = []

        with torch.no_grad():
            for sample in test_samples:
                # Load point cloud
                points = np.load(sample["path"])

                # Normalize (same as training)
                centroid = points.mean(axis=0)
                points = points - centroid

                # Convert to tensor
                points = torch.from_numpy(points).float().unsqueeze(0)  # Add batch dim
                points = points.transpose(1, 2)  # (1, 3, N)
                points = points.to(self.device)

                # Get true label
                species = sample["species"]
                true_label = species_to_idx[species]

                # Forward pass
                logits, embeddings = model(points)
                _, predicted = logits.max(1)
                pred_label = predicted.item()

                # Track results
                is_correct = (pred_label == true_label)
                total += 1
                if is_correct:
                    correct += 1
                    per_species_correct[species] += 1

                per_species_total[species] += 1

                predictions.append({
                    "sample": str(sample["path"]),
                    "species": species,
                    "predicted": list(species_to_idx.keys())[pred_label] if pred_label < len(species_to_idx) else "unknown",
                    "correct": is_correct,
                    "confidence": torch.softmax(logits, dim=1)[0, pred_label].item()
                })

        # Compute metrics
        overall_accuracy = correct / total if total > 0 else 0.0

        per_species_accuracy = {}
        for species in species_to_idx.keys():
            if per_species_total[species] > 0:
                per_species_accuracy[species] = per_species_correct[species] / per_species_total[species]
            else:
                per_species_accuracy[species] = 0.0

        results = {
            "overall_accuracy": overall_accuracy,
            "per_species_accuracy": per_species_accuracy,
            "total_samples": total,
            "correct_samples": correct,
            "predictions": predictions
        }

        return results

    def run_comprehensive_test(
        self,
        model_path: Path,
        data_dir: Path,
        species_list: List[str],
        previous_results: Optional[Dict] = None
    ) -> Tuple[bool, Dict]:
        """Run comprehensive test suite on a model."""
        logger.info(f"Running comprehensive test on model: {model_path}")
        start_time = datetime.now()

        try:
            # Load model
            from src.models.dgcnn import DGCNNHeavy

            checkpoint = torch.load(model_path, map_location=self.device)

            num_classes = len(species_list)
            embedding_dim = checkpoint.get("config", {}).get("embedding_dim", 512)

            model = DGCNNHeavy(num_classes=num_classes, embedding_dim=embedding_dim, k=40)
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to(self.device)

            # Create test set
            test_samples = self.create_test_set(data_dir, species_list)

            if len(test_samples) == 0:
                return False, {"error": "No test samples available"}

            # Create species mapping
            species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}

            # Run test
            test_results = self.test_model(model, test_samples, species_to_idx)

            # Evaluate pass/fail criteria
            passed = True
            failure_reasons = []

            # 1. Check baseline accuracy
            if test_results["overall_accuracy"] < self.baseline_accuracy_threshold:
                passed = False
                failure_reasons.append(
                    f"Overall accuracy {test_results['overall_accuracy']:.3f} below threshold {self.baseline_accuracy_threshold}"
                )

            # 2. Check improvement over previous model (if exists)
            if previous_results is not None:
                prev_accuracy = previous_results.get("overall_accuracy", 0.0)
                improvement = test_results["overall_accuracy"] - prev_accuracy

                if improvement < -0.05:  # Significant degradation
                    passed = False
                    failure_reasons.append(
                        f"Model accuracy degraded by {-improvement:.3f} compared to previous"
                    )
                elif improvement < self.improvement_threshold:
                    logger.warning(
                        f"Model improvement {improvement:.3f} below desired threshold {self.improvement_threshold}"
                    )

            # 3. Check per-species performance (no species should be too bad)
            poor_species = []
            for species, acc in test_results["per_species_accuracy"].items():
                if acc < 0.6:  # Minimum acceptable per-species accuracy
                    poor_species.append((species, acc))

            if poor_species:
                passed = False
                failure_reasons.append(
                    f"Poor accuracy on species: {', '.join([f'{sp}={acc:.2f}' for sp, acc in poor_species])}"
                )

            # Compile final results
            test_duration = (datetime.now() - start_time).total_seconds()

            final_results = {
                "passed": passed,
                "failure_reasons": failure_reasons,
                "test_results": test_results,
                "test_duration_seconds": test_duration,
                "timestamp": datetime.now().isoformat(),
                "model_path": str(model_path),
                "num_test_samples": len(test_samples)
            }

            # Save results
            self._save_test_results(final_results)

            logger.info(
                f"Test {'PASSED' if passed else 'FAILED'}: "
                f"Accuracy={test_results['overall_accuracy']:.3f} ({test_duration:.1f}s)"
            )

            return passed, final_results

        except Exception as e:
            logger.error(f"Test failed with exception: {e}", exc_info=True)
            return False, {"error": str(e), "passed": False}

    def _save_test_results(self, results: Dict):
        """Save test results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.test_results_dir / f"test_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved test results to {results_file}")

    def load_previous_results(self) -> Optional[Dict]:
        """Load most recent test results."""
        result_files = sorted(self.test_results_dir.glob("test_results_*.json"), reverse=True)

        if not result_files:
            return None

        try:
            with open(result_files[0], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load previous results: {e}")
            return None


def compare_models(
    model1_results: Dict,
    model2_results: Dict
) -> Dict:
    """Compare two model test results."""
    acc1 = model1_results["test_results"]["overall_accuracy"]
    acc2 = model2_results["test_results"]["overall_accuracy"]

    improvement = acc2 - acc1
    improvement_pct = (improvement / acc1) * 100 if acc1 > 0 else 0.0

    comparison = {
        "model1_accuracy": acc1,
        "model2_accuracy": acc2,
        "improvement": improvement,
        "improvement_percent": improvement_pct,
        "model2_is_better": acc2 > acc1
    }

    return comparison
