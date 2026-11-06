"""Durable request identities and charging state; no request/response bodies."""
import json
from pathlib import Path
import sqlite3
import time

from .models import Job


class Journal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, "
                "fingerprint TEXT, kind TEXT, model TEXT, state TEXT, "
                "attempts INTEGER, usage TEXT, updated REAL)"
            )
            # A crash may have happened after the provider accepted a request.
            connection.execute("UPDATE jobs SET state='unknown' WHERE state='running'")
            connection.execute("UPDATE jobs SET state='orphan_queued' WHERE state='queued'")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def get(self, ident: str):
        with self.connect() as connection:
            return connection.execute(
                "SELECT fingerprint,state,usage FROM jobs WHERE id=?", (ident,)
            ).fetchone()

    def save(self, job: Job, usage: dict | None = None):
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                (job.ident, job.fingerprint, job.path, job.payload["model"],
                 job.state, len(job.attempts),
                 json.dumps(usage) if usage is not None else None, time.time()),
            )
