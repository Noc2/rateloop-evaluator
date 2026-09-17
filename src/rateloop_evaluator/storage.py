"""Encrypted result cache and durable outbound receipts. Never stores raw inputs."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from contextlib import contextmanager

from cryptography.fernet import Fernet
from .learning import read_secret


class RuntimeStore:
    def __init__(self, path: str | Path, key_file: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.path.is_symlink():
            raise ValueError("Runtime database cannot be a symlink")
        self.cipher = Fernet(read_secret(key_file))
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS results (scope TEXT PRIMARY KEY, digest TEXT NOT NULL, payload BLOB NOT NULL, expires REAL NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, payload BLOB NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0)")
            db.execute("CREATE TABLE IF NOT EXISTS deleted_cases (scope TEXT PRIMARY KEY)")
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("PRAGMA secure_delete=ON")
            with db: yield db
        finally: db.close()

    @staticmethod
    def scope(workspace: str, key: str) -> str:
        return hashlib.sha256(json.dumps([workspace, key]).encode()).hexdigest()

    def encode(self, value: dict) -> bytes:
        return self.cipher.encrypt(json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def decode(self, value: bytes) -> dict:
        return json.loads(self.cipher.decrypt(value))

    def get(self, workspace: str, key: str, digest: str) -> dict | None:
        with self.connect() as db:
            db.execute("DELETE FROM results WHERE expires < ?", (time.time(),))
            row = db.execute("SELECT digest,payload FROM results WHERE scope=?", (self.scope(workspace,key),)).fetchone()
        if row is None: return None
        if row[0] != digest: raise ValueError("Idempotency key is bound to a different input")
        return self.decode(row[1])

    def put(self, workspace: str, key: str, digest: str, result: dict) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_not_deleted(db,result)
            scope = self.scope(workspace,key)
            row = db.execute("SELECT digest,payload FROM results WHERE scope=?", (scope,)).fetchone()
            if row:
                if row[0] != digest: raise ValueError("Idempotency key is bound to a different input")
                return self.decode(row[1])
            db.execute("INSERT INTO results VALUES(?,?,?,?)", (scope,digest,self.encode(result),time.time()+86400))
        return result

    def enqueue(self, receipt_id: str, receipt: dict) -> None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._check_not_deleted(db,receipt)
            row = db.execute("SELECT payload FROM outbox WHERE id=?", (receipt_id,)).fetchone()
            if row and self.decode(row[0]) != receipt: raise ValueError("Outbox ID conflict")
            db.execute("INSERT OR IGNORE INTO outbox(id,payload) VALUES(?,?)", (receipt_id,self.encode(receipt)))

    def pending(self, limit=20) -> list[tuple[str,dict]]:
        with self.connect() as db:
            rows = db.execute("SELECT id,payload FROM outbox WHERE next_attempt <= ? ORDER BY rowid LIMIT ?", (time.time(),limit)).fetchall()
        return [(key,self.decode(payload)) for key,payload in rows]

    def delivered(self, receipt_id: str) -> None:
        with self.connect() as db: db.execute("DELETE FROM outbox WHERE id=?", (receipt_id,))

    def retry(self, receipt_id: str) -> None:
        with self.connect() as db:
            row = db.execute("SELECT attempts FROM outbox WHERE id=?", (receipt_id,)).fetchone()
            if row: db.execute("UPDATE outbox SET attempts=attempts+1,next_attempt=? WHERE id=?", (time.time()+min(3600,2**min(row[0]+1,12)),receipt_id))

    def _check_not_deleted(self, db, payload: dict) -> None:
        result = payload.get("result",payload)
        if result.get("workspaceId") and result.get("caseId"):
            scope = self.scope(result["workspaceId"],result["caseId"])
            if db.execute("SELECT 1 FROM deleted_cases WHERE scope=?",(scope,)).fetchone():
                raise PermissionError("This case was deleted; it cannot be queued or cached again")

    def delete_case(self, workspace: str, case_id: str) -> dict:
        removed = {"results":0,"outbox":0}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO deleted_cases(scope) VALUES(?)",(self.scope(workspace,case_id),))
            for table, key in (("results","scope"),("outbox","id")):
                for row_id, encrypted in db.execute(f"SELECT {key},payload FROM {table}").fetchall():
                    payload = self.decode(encrypted); result = payload.get("result",payload)
                    if result.get("workspaceId") == workspace and result.get("caseId") == case_id:
                        db.execute(f"DELETE FROM {table} WHERE {key}=?",(row_id,)); removed[table] += 1
        return removed
