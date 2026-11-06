"""Adaptive global admission and per-model cooldown policy.

The owning dispatcher holds its condition lock when reading/updating this
object. Cooling requests stay queued and never occupy a consumer.
"""
from email.utils import parsedate_to_datetime
import time


class RateLimiter:
    def __init__(self, initial_interval=0.0, rpm=None):
        if rpm is not None and rpm <= 0:
            raise ValueError("RPM must be positive")
        if initial_interval < 0:
            raise ValueError("Initial interval cannot be negative")
        self.interval = max(initial_interval, 60 / rpm if rpm else 0)
        self.floor = self.interval
        self.global_until = 0.0
        self.model_until = {}
        self.next_start = 0.0
        self._successes = 0

    def ready(self, model, now):
        return now >= max(self.next_start, self.global_until, self.model_until.get(model, 0))

    def admitted(self, now):
        self.next_start = now + self.interval

    def rejected(self, status, model, delay):
        until = time.monotonic() + delay
        if status == 429:
            self.interval = min(60, max(0.25, self.interval * 2))
            self.global_until = max(self.global_until, until)
        else:
            self.model_until[model] = max(self.model_until.get(model, 0), until)

    def succeeded(self):
        self._successes += 1
        if self._successes % 64 == 0:
            self.interval = max(self.floor, self.interval * 0.8)

    @staticmethod
    def retry_after(headers):
        raw = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
        if raw is None:
            return 0.0
        try:
            return max(0.0, float(raw))
        except (ValueError, TypeError):
            try:
                return max(0.0, parsedate_to_datetime(raw).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return 0.0
