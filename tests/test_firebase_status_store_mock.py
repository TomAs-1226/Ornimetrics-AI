import datetime as dt
from types import SimpleNamespace

import pytest

from src.status import firebase_status_store


class DummyDoc:
    def __init__(self):
        self.written = None
        self._collections = {}

    def set(self, payload):
        self.written = payload

    def get(self):
        return SimpleNamespace(exists=True, to_dict=lambda: self.written)

    def collection(self, name):
        if name not in self._collections:
            self._collections[name] = DummyCollection()
        return self._collections[name]


class DummyCollection:
    def __init__(self):
        self.docs = {}

    def document(self, name):
        if name not in self.docs:
            self.docs[name] = DummyDoc()
        return self.docs[name]


class DummyClient:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = DummyCollection()
        return self.collections[name]


def test_write_and_read_uses_firebase_logger(monkeypatch):
    client = DummyClient()
    firestore_module = SimpleNamespace(Client=lambda: client)
    monkeypatch.setattr(firebase_status_store.firebase_logger, "_lazy_import", lambda: (firestore_module, None))

    status = {
        "feeder_id": "abc",
        "state": "OK",
        "revision": 1,
        "updated_at": dt.datetime.utcnow().isoformat(),
    }
    firebase_status_store.write_current_status("abc", status)
    loaded = firebase_status_store.read_last_status("abc")
    assert loaded == status


def test_append_history(monkeypatch):
    client = DummyClient()
    firestore_module = SimpleNamespace(Client=lambda: client)
    monkeypatch.setattr(firebase_status_store.firebase_logger, "_lazy_import", lambda: (firestore_module, None))

    status = {"state": "OK"}
    firebase_status_store.append_history_event("abc", status)
    client = firestore_module.Client()
    history_doc = client.collection("feeders").document("abc").collection("status").document("history")
    assert isinstance(history_doc, DummyDoc)
