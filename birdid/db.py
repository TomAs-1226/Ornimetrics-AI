"""SQLite-backed prototype database for BirdID."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    individual_id: int
    distance: float
    confidence: float


def _serialize_vector(vec: np.ndarray) -> bytes:
    arr = vec.astype(np.float32)
    header = json.dumps({"dtype": "float32", "shape": arr.shape}).encode("utf-8")
    return len(header).to_bytes(4, "big") + header + arr.tobytes()


def _deserialize_vector(blob: bytes) -> np.ndarray:
    header_len = int.from_bytes(blob[:4], "big")
    header = json.loads(blob[4 : 4 + header_len].decode("utf-8"))
    arr = np.frombuffer(blob[4 + header_len :], dtype=np.float32)
    return arr.reshape(header["shape"])


class BirdIDDatabase:
    def __init__(self, path: str = "birdid.sqlite"):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prototypes (
                individual_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                weight REAL DEFAULT 1.0,
                FOREIGN KEY(individual_id) REFERENCES individuals(id)
            );
            """
        )
        self.conn.commit()

    def add_individual(self, species: str, vector: np.ndarray, weight: float = 1.0) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO individuals(species) VALUES (?)", (species,))
        individual_id = cur.lastrowid
        self.add_prototype(individual_id, vector, weight)
        self.conn.commit()
        return int(individual_id)

    def add_prototype(self, individual_id: int, vector: np.ndarray, weight: float = 1.0) -> None:
        blob = _serialize_vector(vector)
        self.conn.execute(
            "INSERT INTO prototypes(individual_id, vector, weight) VALUES (?, ?, ?)",
            (individual_id, blob, float(weight)),
        )
        self.conn.commit()

    def get_prototypes(self, species: str) -> List[Tuple[int, np.ndarray, float]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT prototypes.individual_id, prototypes.vector, prototypes.weight
            FROM prototypes
            JOIN individuals ON individuals.id = prototypes.individual_id
            WHERE individuals.species = ?
            """,
            (species,),
        )
        rows = cur.fetchall()
        return [(int(r[0]), _deserialize_vector(r[1]), float(r[2])) for r in rows]

    def match(self, species: str, embedding: np.ndarray, threshold: float) -> MatchResult:
        prototypes = self.get_prototypes(species)
        if not prototypes:
            new_id = self.add_individual(species, embedding)
            return MatchResult(individual_id=new_id, distance=0.0, confidence=0.0)
        embedding = embedding.astype(np.float32)
        best_id = None
        best_dist = 1e9
        second = 1e9
        for individual_id, proto_vec, weight in prototypes:
            denom = (np.linalg.norm(embedding) * np.linalg.norm(proto_vec)) + 1e-8
            dist = 1 - float(np.dot(embedding, proto_vec) / denom)
            dist /= max(weight, 1e-6)
            if dist < best_dist:
                second = best_dist
                best_dist = dist
                best_id = individual_id
            elif dist < second:
                second = dist
        confidence = max(0.0, min(1.0, (threshold - best_dist) / max(threshold, 1e-6)))
        if best_dist < threshold:
            return MatchResult(individual_id=best_id or 0, distance=best_dist, confidence=confidence)
        new_id = self.add_individual(species, embedding)
        return MatchResult(individual_id=new_id, distance=best_dist, confidence=0.0)

    def update_prototype(self, individual_id: int, new_vec: np.ndarray, ema: float, max_prototypes: int) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT vector, weight, rowid FROM prototypes WHERE individual_id=?", (individual_id,))
        rows = cur.fetchall()
        if not rows:
            self.add_prototype(individual_id, new_vec)
            return
        # Update first prototype with EMA
        blob, weight, rowid = rows[0]
        old_vec = _deserialize_vector(blob)
        merged = (1 - ema) * old_vec + ema * new_vec
        merged /= np.linalg.norm(merged) + 1e-8
        cur.execute("UPDATE prototypes SET vector=?, weight=? WHERE rowid=?", (_serialize_vector(merged), weight, rowid))
        # Keep prototype count bounded
        if len(rows) >= max_prototypes:
            return
        cur.execute("INSERT INTO prototypes(individual_id, vector, weight) VALUES (?, ?, ?)", (individual_id, _serialize_vector(new_vec), weight))
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
