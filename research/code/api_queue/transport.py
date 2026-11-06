"""Provider I/O interface and direct MiniMax implementation."""
import threading
from typing import Protocol

import requests


class Transport(Protocol):
    """Inject a real transport or a test double into the dispatcher."""

    def __call__(self, path: str, payload: dict) -> tuple[int, dict, dict | None]: ...


class DirectTransport:
    """Persistent session per consumer; system/environment proxies disabled."""

    def __init__(self, key: str, endpoint="https://api.minimax.cn/anthropic/v1"):
        self._key = key
        self.endpoint = endpoint
        self._local = threading.local()
        self._sessions = []
        self._lock = threading.Lock()

    def _session(self):
        if not hasattr(self._local, "session"):
            session = requests.Session()
            session.trust_env = False
            session.proxies.clear()
            session.headers.update({
                "Authorization": "Bearer " + self._key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            })
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return self._local.session

    def __call__(self, path, payload):
        timeout = (20, 120 if path.endswith("count_tokens") else 300)
        with self._session().post(
            self.endpoint + path, json=payload, timeout=timeout, allow_redirects=False
        ) as response:
            try:
                data = response.json()
            except ValueError:
                data = None
            return response.status_code, dict(response.headers), data

    def close(self):
        for session in self._sessions:
            session.close()
