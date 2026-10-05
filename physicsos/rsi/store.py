"""Atomic active-strategy transitions; numerical evidence lives in artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from physicsos.schemas.common import ArtifactRef
from physicsos.schemas.rsi import StrategyScope


def canonical_hash(value) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scope_key(scope: StrategyScope) -> str:
    return canonical_hash(scope)


class StrategyStore:
    def __init__(self, workspace: Path):
        self.root = workspace / "data" / "rsi"

    @contextmanager
    def connection(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.root / "state.sqlite3", timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS active (
                    scope TEXT PRIMARY KEY, generation INTEGER NOT NULL,
                    strategy TEXT, activation_id TEXT
                );
                CREATE TABLE IF NOT EXISTS activations (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, generation INTEGER NOT NULL,
                    strategy TEXT NOT NULL, previous_strategy TEXT, previous_activation TEXT,
                    evaluation TEXT NOT NULL, policy TEXT NOT NULL, status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS holdout_uses (
                    fingerprint TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL,
                    selection TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS holdout_problems (
                    identity TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS development_problems (
                    identity TEXT PRIMARY KEY, evaluation_id TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
                    strategy TEXT NOT NULL, identity TEXT NOT NULL, split TEXT NOT NULL,
                    activation_id TEXT, evidence TEXT NOT NULL, assessment TEXT, created_at TEXT NOT NULL
                );
            """)
            if connection.execute("PRAGMA user_version").fetchone()[0] < 1:
                connection.execute("INSERT OR IGNORE INTO development_problems SELECT DISTINCT identity, 'historical-observation' FROM observations WHERE split='development'")
                connection.execute("PRAGMA user_version=1")
                connection.commit()
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def active(self, scope: StrategyScope) -> dict:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM active WHERE scope=?", (scope_key(scope),)).fetchone()
        if row is None:
            return {"generation": 0, "strategy": None, "activation_id": None}
        return {"generation": row["generation"], "strategy": ArtifactRef.model_validate_json(row["strategy"]) if row["strategy"] else None, "activation_id": row["activation_id"]}

    def consume_holdout(self, fingerprint: str, evaluation_id: str, selection: ArtifactRef, identities: list[str]):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._check_holdout(connection, identities)
            connection.execute("INSERT INTO holdout_uses VALUES (?, ?, ?)", (fingerprint, evaluation_id, selection.model_dump_json()))
            connection.executemany("INSERT INTO holdout_problems VALUES (?, ?)", [(identity, evaluation_id) for identity in identities])

    @staticmethod
    def _check_holdout(connection, identities: list[str]):
        if not identities:
            return
        placeholders = ",".join("?" for _ in identities)
        exposed = connection.execute(f"SELECT identity FROM holdout_problems WHERE identity IN ({placeholders}) UNION SELECT identity FROM development_problems WHERE identity IN ({placeholders}) LIMIT 1", (*identities, *identities)).fetchone()
        if exposed:
            raise ValueError("These holdout problems have already been exposed as development or holdout data; register fresh holdout problems.")

    def check_holdout(self, identities: list[str]):
        with self.connection() as connection:
            self._check_holdout(connection, identities)

    def expose_development(self, identities: list[str], evaluation_id: str):
        with self.connection() as connection:
            connection.executemany("INSERT OR IGNORE INTO development_problems VALUES (?, ?)", [(identity, evaluation_id) for identity in identities])

    def holdout_use(self, fingerprint: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM holdout_uses WHERE fingerprint=?", (fingerprint,)).fetchone()
        return dict(row) if row else None

    def promote(self, scope: StrategyScope, expected_generation: int, strategy: ArtifactRef, evaluation: ArtifactRef, policy: dict) -> dict:
        key = scope_key(scope)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM active WHERE scope=?", (key,)).fetchone()
            generation = row["generation"] if row else 0
            if generation != expected_generation:
                raise ValueError("The active strategy changed after this evaluation; evaluate against the current baseline.")
            if row and row["strategy"] == strategy.model_dump_json():
                raise ValueError("This strategy revision is already active.")
            activation_id = uuid4().hex
            generation += 1
            connection.execute("INSERT INTO activations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                activation_id, key, generation, strategy.model_dump_json(),
                row["strategy"] if row else None, row["activation_id"] if row else None,
                evaluation.model_dump_json(), json.dumps(policy), "active",
            ))
            if row and row["activation_id"]:
                connection.execute("UPDATE activations SET status='superseded' WHERE id=?", (row["activation_id"],))
            connection.execute("INSERT OR REPLACE INTO active VALUES (?, ?, ?, ?)", (key, generation, strategy.model_dump_json(), activation_id))
        return {"activation_id": activation_id, "generation": generation}

    def activation(self, activation_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM activations WHERE id=?", (activation_id,)).fetchone()
        return dict(row) if row else None

    def policy_for(self, scope: StrategyScope, strategy: ArtifactRef) -> dict:
        with self.connection() as connection:
            rows = connection.execute("SELECT strategy, policy FROM activations WHERE scope=? ORDER BY generation DESC", (scope_key(scope),)).fetchall()
        return next((json.loads(row["policy"]) for row in rows if ArtifactRef.model_validate_json(row["strategy"]).checksum == strategy.checksum), {})

    def rollback(self, scope: StrategyScope, expected_generation: int, previous: ArtifactRef | None) -> dict:
        key = scope_key(scope)
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM active WHERE scope=?", (key,)).fetchone()
            if row is None or row["generation"] != expected_generation or not row["activation_id"]:
                raise ValueError("No matching active generation is available to roll back.")
            record = connection.execute("SELECT * FROM activations WHERE id=?", (row["activation_id"],)).fetchone()
            if previous and (record["previous_strategy"] is None or ArtifactRef.model_validate_json(record["previous_strategy"]) != previous):
                raise ValueError("Rollback target does not match the recorded predecessor.")
            connection.execute("UPDATE activations SET status='rolled_back' WHERE id=?", (record["id"],))
            restored_activation = record["previous_activation"] if previous else None
            if restored_activation:
                connection.execute("UPDATE activations SET status='active' WHERE id=?", (restored_activation,))
            generation = row["generation"] + 1
            connection.execute("UPDATE active SET generation=?, strategy=?, activation_id=? WHERE scope=?", (generation, previous.model_dump_json() if previous else None, restored_activation, key))
        return {"generation": generation, "activation_id": restored_activation}

    def observe(self, scope: StrategyScope, strategy: ArtifactRef, identity: str, split: str, evidence: ArtifactRef, activation_id: str | None = None, assessment: ArtifactRef | None = None):
        with self.connection() as connection:
            connection.execute("INSERT INTO observations(scope,strategy,identity,split,activation_id,evidence,assessment,created_at) VALUES (?,?,?,?,?,?,?,?)", (
                scope_key(scope), strategy.checksum, identity, split, activation_id,
                evidence.model_dump_json(), assessment.model_dump_json() if assessment else None,
                datetime.now(timezone.utc).isoformat(),
            ))

    def observations(self, scope: StrategyScope, strategy: ArtifactRef) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute("SELECT * FROM observations WHERE scope=? AND strategy=? ORDER BY id", (scope_key(scope), strategy.checksum)).fetchall()
        return [dict(row) for row in rows]
