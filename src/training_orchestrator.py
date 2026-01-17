#!/usr/bin/env python3
"""
Training Orchestrator - Automated Self-Learning System

Coordinates the complete training pipeline:
1. Collects bird data during the day
2. Validates data quality
3. Triggers training at scheduled night hours
4. Tests trained models
5. Deploys successful models
6. Manages model versioning and rollback

Runs autonomously without user intervention.
"""

import threading
import time
import json
import logging
from pathlib import Path
from datetime import datetime, time as dt_time
from typing import Dict, Optional
import pytz

from src.data_validator import DataValidator, TrainingDataset
from src.model_trainer import ModelTrainer
from src.model_tester import ModelTester

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """Orchestrates automated model training pipeline."""

    def __init__(self, config_path: str = "training_config.json"):
        """Initialize orchestrator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.validator = DataValidator(self.config.get("data_quality", {}))
        self.data_dir = Path("data/training_samples")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = TrainingDataset(self.data_dir, self.validator)

        # Training state
        self.is_training = False
        self.training_thread = None
        self.last_training_time = None
        self.training_status = {
            "status": "idle",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "start_time": None,
            "estimated_end_time": None,
            "message": "System ready"
        }

        # Model paths
        self.models_dir = Path("models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_model_path = self.models_dir / "active_model.pth"
        self.previous_model_path = self.models_dir / "previous_model.pth"

        # Timezone
        tz_name = self.config["training"]["schedule"].get("timezone", "America/New_York")
        self.timezone = pytz.timezone(tz_name)

        # Night mode
        self.night_mode_active = False

        logger.info("Training orchestrator initialized")

    def _load_config(self) -> Dict:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            return {"training": {"enabled": False}}

    def _save_config(self):
        """Save current configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def should_train_now(self) -> bool:
        """Check if it's time to start training."""
        if not self.config["training"].get("enabled", False):
            return False

        if self.is_training:
            return False

        # Get current time in configured timezone
        now = datetime.now(self.timezone)
        current_time = now.time()

        # Parse schedule times
        schedule = self.config["training"]["schedule"]
        start_time_str = schedule.get("start_time", "23:00")
        end_time_str = schedule.get("end_time", "05:00")

        start_time = dt_time.fromisoformat(start_time_str)
        end_time = dt_time.fromisoformat(end_time_str)

        # Check if current time is in training window
        if start_time <= end_time:
            # Same day window (e.g., 10:00 - 18:00)
            in_window = start_time <= current_time <= end_time
        else:
            # Overnight window (e.g., 23:00 - 05:00)
            in_window = current_time >= start_time or current_time <= end_time

        if not in_window:
            return False

        # Check if we have enough data
        min_samples = self.config["training"]["data_collection"].get("min_samples_per_species", 50)
        is_ready, info = self.dataset.is_ready_for_training(min_samples)

        if not is_ready:
            logger.debug(f"Not ready for training: {info.get('reason')}")
            return False

        # Check if we already trained today
        if self.last_training_time:
            last_date = self.last_training_time.date()
            current_date = now.date()
            if last_date == current_date:
                logger.debug("Already trained today")
                return False

        return True

    def is_night_time(self) -> bool:
        """Check if it's currently night time (for night mode)."""
        now = datetime.now(self.timezone)
        current_time = now.time()

        schedule = self.config["training"]["schedule"]
        start_time = dt_time.fromisoformat(schedule.get("start_time", "23:00"))
        end_time = dt_time.fromisoformat(schedule.get("end_time", "05:00"))

        if start_time <= end_time:
            return start_time <= current_time <= end_time
        else:
            return current_time >= start_time or current_time <= end_time

    def add_training_sample(
        self,
        points,
        species: str,
        yolo_confidence: float,
        timestamp: float,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Add a validated sample to training dataset."""
        return self.dataset.add_sample(
            points, species, yolo_confidence, timestamp, metadata
        )

    def start_training_async(self):
        """Start training in background thread."""
        if self.is_training:
            logger.warning("Training already in progress")
            return

        self.training_thread = threading.Thread(
            target=self._training_pipeline,
            daemon=True
        )
        self.training_thread.start()
        logger.info("Started training pipeline in background")

    def _training_pipeline(self):
        """Complete training pipeline."""
        try:
            self.is_training = True
            self.training_status = {
                "status": "preparing",
                "progress": 0.0,
                "message": "Preparing training data...",
                "start_time": datetime.now().isoformat()
            }

            logger.info("=" * 60)
            logger.info("STARTING AUTOMATED TRAINING PIPELINE")
            logger.info("=" * 60)

            # 1. Prepare data
            logger.info("Step 1/5: Preparing training data")
            species_counts = self.dataset.get_species_count()
            species_list = list(species_counts.keys())

            if not species_list:
                raise ValueError("No species data available")

            logger.info(f"Training on {len(species_list)} species: {species_list}")
            for species, count in species_counts.items():
                logger.info(f"  - {species}: {count} samples")

            # 2. Initialize trainer
            logger.info("Step 2/5: Initializing model trainer")
            self.training_status["status"] = "initializing"
            self.training_status["progress"] = 20.0

            trainer_config = self.config["training"]["model"]
            trainer = ModelTrainer(trainer_config, device="cpu")

            # Prepare data
            val_split = self.config["training"]["data_collection"].get("validation_split", 0.2)
            train_loader, val_loader = trainer.prepare_data(self.data_dir, species_list, val_split)

            # Initialize model
            embedding_dim = trainer_config.get("embedding_dim", 512)
            trainer.initialize_model(num_classes=len(species_list), embedding_dim=embedding_dim)

            # 3. Train model
            logger.info("Step 3/5: Training model")
            self.training_status["status"] = "training"
            self.training_status["progress"] = 30.0
            self.training_status["total_epochs"] = trainer.epochs

            # Wrap training to update progress
            original_train_epoch = trainer.train_epoch

            def tracked_train_epoch(train_loader):
                result = original_train_epoch(train_loader)
                self.training_status["current_epoch"] = len(trainer.training_history["epochs"])
                self.training_status["progress"] = 30.0 + (
                    50.0 * self.training_status["current_epoch"] / trainer.epochs
                )
                return result

            trainer.train_epoch = tracked_train_epoch

            # Run training
            training_results = trainer.train(train_loader, val_loader)

            logger.info(f"Training completed: {training_results}")

            # 4. Test model
            logger.info("Step 4/5: Testing trained model")
            self.training_status["status"] = "testing"
            self.training_status["progress"] = 80.0

            tester_config = self.config["training"]["validation"]
            tester = ModelTester(tester_config, device="cpu")

            # Load previous test results for comparison
            previous_results = tester.load_previous_results()

            # Test new model
            best_model_path = trainer.checkpoint_dir / "best_model.pth"
            test_passed, test_results = tester.run_comprehensive_test(
                best_model_path,
                self.data_dir,
                species_list,
                previous_results
            )

            if not test_passed:
                logger.error("Model failed validation tests")
                logger.error(f"Failure reasons: {test_results.get('failure_reasons')}")

                if self.config["training"]["deployment"].get("auto_deploy_on_pass", True):
                    raise ValueError("Model failed validation - not deploying")
            else:
                logger.info(f"Model passed all tests! Accuracy: {test_results['test_results']['overall_accuracy']:.3f}")

            # 5. Deploy model
            logger.info("Step 5/5: Deploying model")
            self.training_status["status"] = "deploying"
            self.training_status["progress"] = 90.0

            self._deploy_model(best_model_path, test_results)

            # Success!
            self.training_status = {
                "status": "completed",
                "progress": 100.0,
                "message": "Training completed successfully",
                "end_time": datetime.now().isoformat(),
                "results": {
                    "accuracy": test_results["test_results"]["overall_accuracy"],
                    "training_time_hours": training_results["training_time_hours"],
                    "final_epoch": training_results["final_epoch"]
                }
            }

            self.last_training_time = datetime.now(self.timezone)

            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}", exc_info=True)
            self.training_status = {
                "status": "failed",
                "progress": 0.0,
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "end_time": datetime.now().isoformat()
            }

        finally:
            self.is_training = False

    def _deploy_model(self, model_path: Path, test_results: Dict):
        """Deploy a validated model."""
        # Backup previous model if exists
        if self.active_model_path.exists():
            logger.info("Backing up previous model")
            if self.previous_model_path.exists():
                self.previous_model_path.unlink()  # Remove old backup
            self.active_model_path.rename(self.previous_model_path)

        # Copy new model to active
        import shutil
        shutil.copy(model_path, self.active_model_path)

        # Save deployment info
        deployment_info = {
            "deployed_at": datetime.now().isoformat(),
            "model_path": str(model_path),
            "test_results": test_results,
            "version": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        deployment_file = self.models_dir / "deployment_info.json"
        with open(deployment_file, 'w') as f:
            json.dump(deployment_info, f, indent=2)

        logger.info(f"Deployed new model: {self.active_model_path}")

    def rollback_model(self) -> bool:
        """Rollback to previous model version."""
        if not self.previous_model_path.exists():
            logger.error("No previous model available for rollback")
            return False

        try:
            # Swap models
            temp_path = self.models_dir / "temp_rollback.pth"
            self.active_model_path.rename(temp_path)
            self.previous_model_path.rename(self.active_model_path)
            temp_path.rename(self.previous_model_path)

            logger.info("Rolled back to previous model")
            return True

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False

    def get_status(self) -> Dict:
        """Get current training system status."""
        species_counts = self.dataset.get_species_count()
        is_ready, ready_info = self.dataset.is_ready_for_training(
            self.config["training"]["data_collection"].get("min_samples_per_species", 50)
        )

        return {
            "training_enabled": self.config["training"].get("enabled", False),
            "is_training": self.is_training,
            "night_mode_active": self.night_mode_active,
            "should_train_now": self.should_train_now(),
            "is_night_time": self.is_night_time(),
            "training_status": self.training_status,
            "dataset_ready": is_ready,
            "dataset_info": ready_info,
            "species_counts": species_counts,
            "active_model_exists": self.active_model_path.exists(),
            "previous_model_exists": self.previous_model_path.exists(),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "schedule": self.config["training"]["schedule"]
        }

    def run_monitoring_loop(self):
        """Background monitoring loop (runs in detection system)."""
        logger.info("Training orchestrator monitoring loop started")

        while True:
            try:
                # Check if it's training time
                if self.should_train_now() and not self.is_training:
                    logger.info("Training time detected - starting automated training")
                    self.start_training_async()

                # Update night mode status
                self.night_mode_active = self.is_night_time()

                # Sleep for a bit
                time.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error
