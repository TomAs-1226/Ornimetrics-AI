"""Firebase-backed status store for live feeder state.

This module reuses the existing ``birdid.firebase_logger`` lazy Firebase
initialisation to avoid duplicating credential handling or altering existing
event logging behaviour.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, Optional

from birdid import firebase_logger


def _get_firestore_client():
    firestore, _ = firebase_logger._lazy_import()
    if firestore is None:
        raise RuntimeError("Firebase client libraries not available")
    return firestore.Client()


def write_current_status(feeder_id: str, status: Dict[str, Any], collection: str = "feeders") -> None:
    """Write the current status atomically to Firestore."""

    client = _get_firestore_client()
    doc_ref = client.collection(collection).document(feeder_id).collection("status").document("current")
    doc_ref.set(status)


def append_history_event(
    feeder_id: str,
    status: Dict[str, Any],
    collection: str = "feeders",
    timestamp: Optional[dt.datetime] = None,
) -> None:
    """Append a history entry under a revision or timestamped document."""

    client = _get_firestore_client()
    history_ref = (
        client.collection(collection)
        .document(feeder_id)
        .collection("status")
        .document("history")
        .collection("events")
    )
    ts = timestamp or dt.datetime.utcnow()
    history_ref.document(ts.isoformat()).set(status)


def read_last_status(feeder_id: str, collection: str = "feeders") -> Optional[Dict[str, Any]]:
    firestore, _ = firebase_logger._lazy_import()
    if firestore is None:
        return None
    client = firestore.Client()
    doc = client.collection(collection).document(feeder_id).collection("status").document("current").get()
    if not doc.exists:
        return None
    return doc.to_dict()

