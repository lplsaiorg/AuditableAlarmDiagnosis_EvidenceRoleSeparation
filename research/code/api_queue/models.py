"""Queue request and failure types, independent of HTTP and persistence."""
from concurrent.futures import Future
from dataclasses import dataclass, field
import time

MODELS = frozenset({"MiniMax-M3", "MiniMax-M3.1-Flash-Preview", "MiniMax-M3.1-Flash"})
PATHS = frozenset({"/messages", "/messages/count_tokens", "/chat/completions"})


class DispatchError(RuntimeError):
    """A rejected/failed request; unknown means charging cannot be determined."""

    def __init__(self, message: str, *, unknown: bool = False):
        super().__init__(message)
        self.unknown = unknown


@dataclass
class Job:
    """In-memory payload and completion handle; bodies never enter the journal."""

    ident: str
    path: str
    payload: dict
    fingerprint: str
    future: Future = field(default_factory=Future)
    attempts: list = field(default_factory=list)
    submitted: float = field(default_factory=time.monotonic)
    state: str = "queued"
    finished: float | None = None

    @property
    def counting(self) -> bool:
        return self.path == "/messages/count_tokens"


@dataclass
class AttemptResult:
    """Consumer outcome passed back to the synchronized queue state."""

    record: dict
    data: dict | None = None
    error: str | None = None
    unknown: bool = False
    retry: bool = False
