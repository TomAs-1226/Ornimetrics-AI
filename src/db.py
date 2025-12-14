import sqlite3
from typing import Optional, Tuple
import numpy as np


class IdentityDB:
    def __init__(self, path: str = "individuals.db"):
        self.conn = sqlite3.connect(path)
        self._ensure_tables()

    def _ensure_tables(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                individual_id TEXT UNIQUE,
                name TEXT,
                class TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                individual_id TEXT,
                embedding BLOB,
                FOREIGN KEY(individual_id) REFERENCES individuals(individual_id)
            );
            """
        )
        self.conn.commit()

    def upsert_individual(self, individual_id: str, name: str, class_name: str, embedding: Optional[np.ndarray] = None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO individuals(individual_id, name, class) VALUES(?,?,?)",
            (individual_id, name, class_name),
        )
        if embedding is not None:
            cur.execute(
                "INSERT INTO embeddings(individual_id, embedding) VALUES(?, ?)",
                (individual_id, embedding.astype("float32").tobytes()),
            )
        self.conn.commit()

    def get_identity(self, individual_id: str) -> Optional[Tuple[str, str]]:
        cur = self.conn.cursor()
        cur.execute("SELECT name, class FROM individuals WHERE individual_id=?", (individual_id,))
        row = cur.fetchone()
        return (row[0], row[1]) if row else None

    def load_embeddings(self, individual_id: str) -> Optional[np.ndarray]:
        cur = self.conn.cursor()
        cur.execute("SELECT embedding FROM embeddings WHERE individual_id=?", (individual_id,))
        rows = cur.fetchall()
        if not rows:
            return None
        embs = [np.frombuffer(r[0], dtype="float32") for r in rows]
        return np.stack(embs)

    def close(self):
        self.conn.close()

