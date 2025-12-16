from types import SimpleNamespace

from birdid import firebase_logger


class DummyCollection:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)


class DummyClient:
    def __init__(self):
        self.collection_obj = DummyCollection()

    def collection(self, name):
        return self.collection_obj


def test_log_event_unchanged(monkeypatch):
    client = DummyClient()
    firestore_module = SimpleNamespace(Client=lambda: client)
    monkeypatch.setattr(firebase_logger, "_lazy_import", lambda: (firestore_module, None))

    firebase_logger.log_event({"foo": "bar"})
    assert client.collection_obj.events[0]["foo"] == "bar"
