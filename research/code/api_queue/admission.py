"""Adapter for sharing the global concurrency cap with legacy workers."""
import os
import time
from typing import Protocol
import uuid


class SlotManager(Protocol):
    def try_acquire(self) -> str | None: ...
    def release(self, ident: str) -> None: ...
    def rate_limited(self, header: str) -> None: ...


class LegacySlots:
    """Share existing leases/cooldown without the legacy fixed request gap.

    The SharedGate dependency is confined to this adapter. Both old callers and
    queue consumers count toward the same hard limit of four live requests.
    """

    def __init__(self, path=None, *, max_concurrency=4):
        from shared_api_gate import SharedGate
        if max_concurrency not in (4, 6):
            raise ValueError("Use the default cap 4 or explicit experimental cap 6")
        self.max_concurrency = max_concurrency
        self.gate = SharedGate(path)

    def try_acquire(self):
        from shared_api_gate import alive
        now = time.time()
        with self.gate.db() as connection:
            expired = connection.execute(
                "SELECT id,pid FROM leases WHERE expires<=?", (now,)
            ).fetchall()
            for ident, pid in expired:
                if not alive(pid):
                    connection.execute("DELETE FROM leases WHERE id=?", (ident,))
            cooldown = connection.execute(
                "SELECT cooldown FROM state WHERE id=1"
            ).fetchone()[0]
            occupied = connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0]
            if cooldown > now or occupied >= self.max_concurrency:
                return None
            ident = uuid.uuid4().hex
            connection.execute("INSERT INTO leases VALUES (?,?,?)", (ident, os.getpid(), now + 600))
            connection.execute("INSERT INTO events VALUES (?,?,?)", (now, "dispatcher_admitted", ident))
            return ident

    def release(self, ident):
        self.gate.release(ident)

    def rate_limited(self, header):
        self.gate.rate_limited(header)
