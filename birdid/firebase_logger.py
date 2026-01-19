"""Optional Firebase logging helpers for Ornimetrics OS.

This module intentionally keeps Firebase as an optional dependency so that the
Pi runtime stays lean. On a PC demo, set ``GOOGLE_APPLICATION_CREDENTIALS`` to a
service account JSON and provide the appropriate project configuration.

Ornimetrics OS Update:
- User-based data paths: /users/{user_id}/feeders/{device_id}/
- Account linking required before data upload
- Privacy controls (user sees only their own data)
- Data publishing toggle for future app functionality
"""
from __future__ import annotations

import logging
import os
import json
from pathlib import Path
from typing import Dict, Optional


LOGGER = logging.getLogger(__name__)

_firestore = None
_storage = None
_ornimetrics_config = None


def _lazy_import() -> tuple[object | None, object | None]:
    global _firestore, _storage
    if _firestore is not None or _storage is not None:
        return _firestore, _storage
    try:
        from google.cloud import firestore  # type: ignore
        from google.cloud import storage  # type: ignore
    except Exception:
        LOGGER.warning("Firebase client libraries not available; running local-only")
        _firestore, _storage = None, None
        return _firestore, _storage
    _firestore = firestore
    _storage = storage
    return _firestore, _storage


def _load_ornimetrics_config() -> Dict:
    """Load Ornimetrics OS configuration."""
    global _ornimetrics_config
    if _ornimetrics_config is not None:
        return _ornimetrics_config

    config_path = Path("ornimetrics_os_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            _ornimetrics_config = json.load(f)
            return _ornimetrics_config

    # Fallback to legacy mode
    _ornimetrics_config = {
        "account": {"linked": False},
        "firebase": {"structure": "legacy", "data_publishing_enabled": True}
    }
    return _ornimetrics_config


def is_available() -> bool:
    firestore, _ = _lazy_import()
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    return firestore is not None and creds and os.path.exists(creds)


def is_account_linked() -> bool:
    """Check if device is linked to a user account."""
    config = _load_ornimetrics_config()
    return config.get("account", {}).get("linked", False)


def is_data_publishing_enabled() -> bool:
    """Check if data publishing is enabled."""
    config = _load_ornimetrics_config()
    return config.get("firebase", {}).get("data_publishing_enabled", True)


def get_user_based_path(collection: str = "detections") -> str:
    """Get user-based Firestore path for data."""
    config = _load_ornimetrics_config()

    # Check if using user-based structure
    if config.get("firebase", {}).get("structure") != "user_based":
        # Legacy mode - use old path
        return collection

    # Check if account is linked
    if not is_account_linked():
        LOGGER.warning("Account not linked - data upload disabled")
        return None

    # Get user and device IDs
    user_id = config.get("account", {}).get("user_id")
    device_id = config.get("system", {}).get("device_id")

    if not user_id or not device_id:
        LOGGER.error("Missing user_id or device_id in configuration")
        return None

    # Build user-based path: /users/{user_id}/feeders/{device_id}/detections
    path = f"users/{user_id}/feeders/{device_id}/{collection}"
    return path


def log_event(event: Dict, collection: str = "detections") -> None:
    """Log event to Firestore with user-based path support.

    Args:
        event: Event data to log
        collection: Collection name (will be under user/feeder path if configured)
    """
    firestore, _ = _lazy_import()
    if firestore is None:
        return

    # Check if data publishing is enabled
    if not is_data_publishing_enabled():
        LOGGER.debug("Data publishing disabled - skipping event log")
        return

    # Get appropriate path (user-based or legacy)
    path = get_user_based_path(collection)
    if path is None:
        LOGGER.warning("Cannot determine Firebase path - skipping event log")
        return

    try:
        client = firestore.Client()

        # Add Ornimetrics OS metadata
        event["_ornimetrics_os"] = {
            "version": _load_ornimetrics_config().get("_branding", {}).get("version", "1.0.0"),
            "device_id": _load_ornimetrics_config().get("system", {}).get("device_id"),
            "feeder_name": _load_ornimetrics_config().get("account", {}).get("feeder_name", "Unknown")
        }

        # Use subcollection structure for user-based paths
        if "/" in path:
            # Path like: users/{user_id}/feeders/{device_id}/detections
            # Navigate through document/collection hierarchy
            parts = path.split("/")
            ref = client.collection(parts[0])

            for i in range(1, len(parts)):
                if i % 2 == 1:  # Document
                    ref = ref.document(parts[i])
                else:  # Collection
                    ref = ref.collection(parts[i])

            ref.add(event)
        else:
            # Legacy flat collection
            client.collection(path).add(event)

        LOGGER.debug(f"Event logged to Firebase: {path}")

    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Failed to log event to Firebase: %s", exc)


def upload_artifacts(
    image_bytes: Optional[bytes],
    pointcloud_bytes: Optional[bytes],
    metadata: Dict,
    bucket_name: Optional[str] = None,
    image_path: str = "birdid/images/",  # prefix (legacy)
    point_path: str = "birdid/pointclouds/",  # prefix (legacy)
) -> None:
    """Upload detection artifacts to Cloud Storage with user-based paths.

    Args:
        image_bytes: Image data
        pointcloud_bytes: Point cloud data
        metadata: Detection metadata
        bucket_name: Storage bucket name (optional)
        image_path: Legacy image path prefix
        point_path: Legacy point cloud path prefix
    """
    firestore, storage = _lazy_import()
    if storage is None:
        return

    # Check if data publishing is enabled
    if not is_data_publishing_enabled():
        LOGGER.debug("Data publishing disabled - skipping artifact upload")
        return

    # Get configuration
    config = _load_ornimetrics_config()
    user_based = config.get("firebase", {}).get("structure") == "user_based"

    # Check account linking for user-based structure
    if user_based and not is_account_linked():
        LOGGER.warning("Account not linked - artifact upload disabled")
        return

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name) if bucket_name else client.get_default_bucket()

        # Determine path prefix based on structure
        if user_based:
            user_id = config.get("account", {}).get("user_id")
            device_id = config.get("system", {}).get("device_id")

            if not user_id or not device_id:
                LOGGER.error("Missing user_id or device_id - skipping upload")
                return

            # User-based paths: users/{user_id}/feeders/{device_id}/images/
            image_prefix = f"users/{user_id}/feeders/{device_id}/images/"
            point_prefix = f"users/{user_id}/feeders/{device_id}/pointclouds/"
        else:
            # Legacy flat paths
            image_prefix = image_path
            point_prefix = point_path

        # Upload image
        if image_bytes is not None:
            blob = bucket.blob(f"{image_prefix}{metadata.get('run_id','run')}.png")
            blob.upload_from_string(image_bytes, content_type="image/png")
            LOGGER.debug(f"Uploaded image to: {image_prefix}")

        # Upload point cloud
        if pointcloud_bytes is not None:
            blob = bucket.blob(f"{point_prefix}{metadata.get('run_id','run')}.ply")
            blob.upload_from_string(pointcloud_bytes, content_type="application/octet-stream")
            LOGGER.debug(f"Uploaded point cloud to: {point_prefix}")

        # Log metadata event
        if firestore is not None:
            log_event(metadata, collection="detections")

    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Failed to upload artifacts to Firebase: %s", exc)

