"""Optional Firebase logging helpers.

This module intentionally keeps Firebase as an optional dependency so that the
Pi runtime stays lean. On a PC demo, set ``GOOGLE_APPLICATION_CREDENTIALS`` to a
service account JSON and provide the appropriate project configuration.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional


LOGGER = logging.getLogger(__name__)

_firestore = None
_storage = None


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


def is_available() -> bool:
    firestore, _ = _lazy_import()
    creds = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    return firestore is not None and creds and os.path.exists(creds)


def log_event(event: Dict, collection: str = "birdid_events") -> None:
    firestore, _ = _lazy_import()
    if firestore is None:
        return
    try:
        client = firestore.Client()
        client.collection(collection).add(event)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Failed to log event to Firebase: %s", exc)


def upload_artifacts(
    image_bytes: Optional[bytes],
    pointcloud_bytes: Optional[bytes],
    metadata: Dict,
    bucket_name: Optional[str] = None,
    image_path: str = "birdid/images/",  # prefix
    point_path: str = "birdid/pointclouds/",
) -> None:
    firestore, storage = _lazy_import()
    if storage is None:
        return
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name) if bucket_name else client.get_default_bucket()
        if image_bytes is not None:
            blob = bucket.blob(f"{image_path}{metadata.get('run_id','run')}.png")
            blob.upload_from_string(image_bytes, content_type="image/png")
        if pointcloud_bytes is not None:
            blob = bucket.blob(f"{point_path}{metadata.get('run_id','run')}.ply")
            blob.upload_from_string(pointcloud_bytes, content_type="application/octet-stream")
        if firestore is not None:
            log_event(metadata)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Failed to upload artifacts to Firebase: %s", exc)

