"""SQLite-backed prototype database for BirdID."""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class MatchResult:
    individual_id: int | None
    distance: float
    confidence: float
    second_best: float
    is_match: bool
    has_prototypes: bool


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

    def _add_column_if_missing(self, table: str, column: str, decl: str) -> None:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
            self.conn.commit()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS individuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                last_seen_ts REAL DEFAULT 0,
                last_refresh_ts REAL DEFAULT 0
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prototypes (
                individual_id INTEGER NOT NULL,
                vector BLOB NOT NULL,
                weight REAL DEFAULT 1.0,
                created_ts REAL DEFAULT 0,
                FOREIGN KEY(individual_id) REFERENCES individuals(id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS individual_stats (
                individual_id INTEGER PRIMARY KEY,
                last_dispense_ts REAL DEFAULT 0,
                dispense_count_today INTEGER DEFAULT 0,
                attempt_count_today INTEGER DEFAULT 0,
                last_reset_date TEXT
            );
            """
        )
        self.conn.commit()
        # Migration for older installs
        self._add_column_if_missing("individuals", "last_seen_ts", "REAL DEFAULT 0")
        self._add_column_if_missing("individuals", "last_refresh_ts", "REAL DEFAULT 0")
        self._add_column_if_missing("prototypes", "created_ts", "REAL DEFAULT 0")

    def add_individual(self, species: str, vector: np.ndarray, weight: float = 1.0) -> int:
        cur = self.conn.cursor()
        now_ts = time.time()
        cur.execute("INSERT INTO individuals(species, last_seen_ts, last_refresh_ts) VALUES (?, ?, ?)", (species, now_ts, now_ts))
        individual_id = cur.lastrowid
        self.add_prototype(individual_id, vector, weight, created_ts=now_ts)
        self._ensure_stats_row(individual_id)
        self.conn.commit()
        return int(individual_id)

    def add_prototype(self, individual_id: int, vector: np.ndarray, weight: float = 1.0, created_ts: float | None = None) -> None:
        blob = _serialize_vector(vector)
        created_ts = time.time() if created_ts is None else created_ts
        self.conn.execute(
            "INSERT INTO prototypes(individual_id, vector, weight, created_ts) VALUES (?, ?, ?, ?)",
            (individual_id, blob, float(weight), float(created_ts)),
        )
        self.conn.commit()

    def _ensure_stats_row(self, individual_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM individual_stats WHERE individual_id=?", (individual_id,))
        if cur.fetchone() is None:
            cur.execute(
                "INSERT INTO individual_stats(individual_id, last_dispense_ts, dispense_count_today, attempt_count_today, last_reset_date)"
                " VALUES (?, 0, 0, 0, NULL)",
                (individual_id,),
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
            return MatchResult(individual_id=None, distance=float("inf"), confidence=0.0, second_best=float("inf"), is_match=False, has_prototypes=False)
        embedding = embedding.astype(np.float32)
        best_id = None
        best_dist = float("inf")
        second = float("inf")
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
        is_match = best_dist < threshold
        return MatchResult(individual_id=best_id, distance=best_dist, confidence=confidence, second_best=second, is_match=is_match, has_prototypes=True)

    def update_prototype(self, individual_id: int, new_vec: np.ndarray, ema: float, max_prototypes: int, now_ts: float | None = None) -> None:
        cur = self.conn.cursor()
        cur.execute("SELECT vector, weight, rowid FROM prototypes WHERE individual_id=?", (individual_id,))
        rows = cur.fetchall()
        if not rows:
            self.add_prototype(individual_id, new_vec, created_ts=now_ts)
            return
        # Update first prototype with EMA
        blob, weight, rowid = rows[0]
        old_vec = _deserialize_vector(blob)
        merged = (1 - ema) * old_vec + ema * new_vec
        merged /= np.linalg.norm(merged) + 1e-8
        cur.execute("UPDATE prototypes SET vector=?, weight=? WHERE rowid=?", (_serialize_vector(merged), weight, rowid))
        if now_ts is not None:
            self.mark_refreshed(individual_id, now_ts)
        # Keep prototype count bounded
        if len(rows) >= max_prototypes:
            return
        cur.execute("INSERT INTO prototypes(individual_id, vector, weight, created_ts) VALUES (?, ?, ?, ?)", (individual_id, _serialize_vector(new_vec), weight, float(now_ts or time.time())))
        self.conn.commit()
        if now_ts is not None:
            self.mark_refreshed(individual_id, now_ts)

    def mark_seen(self, individual_id: int, now_ts: float) -> None:
        self.conn.execute("UPDATE individuals SET last_seen_ts=? WHERE id=?", (now_ts, individual_id))
        self.conn.commit()

    def last_refresh_ts(self, individual_id: int) -> float:
        cur = self.conn.cursor()
        cur.execute("SELECT last_refresh_ts FROM individuals WHERE id=?", (individual_id,))
        row = cur.fetchone()
        return float(row[0] or 0.0) if row else 0.0

    def is_refresh_stale(self, individual_id: int, now_ts: float, days: int) -> bool:
        last = self.last_refresh_ts(individual_id)
        return (now_ts - last) >= days * 86400

    def mark_refreshed(self, individual_id: int, now_ts: float) -> None:
        self.conn.execute("UPDATE individuals SET last_refresh_ts=? WHERE id=?", (now_ts, individual_id))
        self.conn.commit()

    def _reset_stats_if_needed(self, individual_id: int, today: str) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT last_reset_date FROM individual_stats WHERE individual_id=?",
            (individual_id,),
        )
        row = cur.fetchone()
        if row is None:
            self._ensure_stats_row(individual_id)
            return
        last_reset = row[0]
        if last_reset != today:
            cur.execute(
                "UPDATE individual_stats SET dispense_count_today=0, attempt_count_today=0, last_reset_date=? WHERE individual_id=?",
                (today, individual_id),
            )
            self.conn.commit()

    def get_stats(self, individual_id: int, now_ts: float) -> Dict[str, float | int]:
        today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
        self._reset_stats_if_needed(individual_id, today)
        cur = self.conn.cursor()
        cur.execute(
            "SELECT last_dispense_ts, dispense_count_today, attempt_count_today, last_reset_date FROM individual_stats WHERE individual_id=?",
            (individual_id,),
        )
        row = cur.fetchone()
        if row is None:
            self._ensure_stats_row(individual_id)
            return {"last_dispense_ts": 0.0, "dispense_count_today": 0, "attempt_count_today": 0}
        return {
            "last_dispense_ts": float(row[0] or 0.0),
            "dispense_count_today": int(row[1] or 0),
            "attempt_count_today": int(row[2] or 0),
        }

    def record_attempt(self, individual_id: int, now_ts: float) -> None:
        today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
        self._reset_stats_if_needed(individual_id, today)
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE individual_stats SET attempt_count_today=attempt_count_today+1, last_reset_date=? WHERE individual_id=?",
            (today, individual_id),
        )
        self.conn.commit()

    def record_dispense(self, individual_id: int, now_ts: float) -> None:
        today = time.strftime("%Y-%m-%d", time.localtime(now_ts))
        self._reset_stats_if_needed(individual_id, today)
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE individual_stats SET attempt_count_today=attempt_count_today+1, dispense_count_today=dispense_count_today+1, last_dispense_ts=?, last_reset_date=? WHERE individual_id=?",
            (now_ts, today, individual_id),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()
