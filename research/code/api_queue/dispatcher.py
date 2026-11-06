"""Bounded producer/consumer queue, caching, idempotency and request lifecycle."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
import uuid

from .admission import SlotManager
from .journal import Journal
from .models import AttemptResult, DispatchError, Job, MODELS, PATHS
from .rate_limit import RateLimiter
from .transport import Transport


class Dispatcher:
    """Own consumer threads; inject provider I/O, lease storage and rate policy.

    Producers submit jobs and wait on their futures. Only consumers perform
    provider I/O. Generation identity is explicit; equal generation payloads
    with different IDs remain separate billable requests.
    """

    def __init__(
        self, transport: Transport, *, workers=4, capacity=64,
        journal: Journal | str | Path | None = None,
        slots: SlotManager | None = None, initial_interval=0.0, rpm=None,
        max_attempts=8, cache_size=512, result_ttl=900,
        rate_limiter: RateLimiter | None = None,
        max_concurrency=4,
    ):
        pooled = hasattr(slots, 'dispatch') and max_concurrency == getattr(slots, 'max_concurrency', None)
        if (max_concurrency not in (4, 6) and not pooled) or not 1 <= workers <= max_concurrency:
            raise ValueError("Consumers must fit the default cap 4 or explicit experimental cap 6")
        if capacity < workers:
            raise ValueError("Queue capacity must cover active consumers")
        self.transport = transport
        self.capacity = capacity
        self.slots = slots
        self.journal = Journal(journal) if isinstance(journal, (str, Path)) else journal
        self.rate_limiter = rate_limiter or RateLimiter(initial_interval, rpm)
        self.cv = threading.Condition()
        self.pending = []
        self.jobs = OrderedDict()
        self.counts = OrderedDict()
        self.count_flights = {}
        self.active = 0
        self.incomplete = 0
        self.closed = False
        self.max_attempts = max_attempts
        self.cache_size = cache_size
        self.ttl = result_ttl
        self.metrics = {
            "provider_requests": 0, "count_cache_hits": 0, "coalesced_counts": 0,
            "peak_active": 0, "completed": 0, "unknown": 0,
        }
        self.threads = [
            threading.Thread(target=self._consumer, name=f"api-consumer-{i}", daemon=True)
            for i in range(workers)
        ]
        for thread in self.threads:
            thread.start()

    @staticmethod
    def fingerprint(path, payload):
        encoded = path + "\n" + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _prune(self):
        now = time.monotonic()
        for ident, job in list(self.jobs.items()):
            if job.future.done() and job.finished is not None and now - job.finished > self.ttl:
                del self.jobs[ident]

    def _check_journal(self, ident, fingerprint):
        if self.journal is None:
            return
        old = self.journal.get(ident)
        if old and old[0] != fingerprint:
            raise DispatchError("Request ID conflicts with durable journal")
        if old and old[1] != "orphan_queued":
            raise DispatchError(
                "Recorded request response is unavailable; do not resend generation",
                unknown=old[1] in ["unknown", "running", "succeeded"],
            )

    def _ensure_capacity(self):
        if self.incomplete >= self.capacity:
            raise DispatchError("Queue is full; producer must apply backpressure")
        if len(self.jobs) >= self.capacity + self.cache_size:
            for ident, job in list(self.jobs.items()):
                if job.future.done():
                    del self.jobs[ident]
                    break
            else:
                raise DispatchError("Queue is full; producer must apply backpressure")

    def submit(self, path: str, payload: dict, ident: str | None = None) -> Job:
        if path not in PATHS or payload.get("model") not in MODELS:
            raise ValueError("Unsupported endpoint/model")
        # Detach ownership: producers cannot mutate an admitted payload.
        payload = json.loads(json.dumps(payload))
        ident = ident or uuid.uuid4().hex
        fingerprint = self.fingerprint(path, payload)
        with self.cv:
            if self.closed:
                raise DispatchError("Dispatcher is closed")
            self._prune()
            if ident in self.jobs:
                job = self.jobs[ident]
                if fingerprint != job.fingerprint:
                    raise DispatchError("Request ID reused with a different payload")
                return job
            self._check_journal(ident, fingerprint)
            self._ensure_capacity()
            if path == "/messages/count_tokens":
                if fingerprint in self.count_flights:
                    job = self.count_flights[fingerprint]
                    self.jobs[ident] = job
                    self.metrics["coalesced_counts"] += 1
                    return job
                if fingerprint in self.counts:
                    job = Job(ident, path, payload, fingerprint)
                    job.state = "succeeded"
                    job.finished = time.monotonic()
                    job.future.set_result(self.counts[fingerprint])
                    self.jobs[ident] = job
                    self.metrics["count_cache_hits"] += 1
                    return job
            job = Job(ident, path, payload, fingerprint)
            # Persist admission before making work visible to consumers.
            if self.journal:
                self.journal.save(job)
            self.jobs[ident] = job
            self.pending.append(job)
            self.incomplete += 1
            if job.counting:
                self.count_flights[fingerprint] = job
            self.cv.notify_all()
            return job

    def get(self, ident: str) -> Job | None:
        with self.cv:
            return self.jobs.get(ident)

    def _take(self):
        with self.cv:
            while True:
                if self.closed and not self.pending:
                    return None
                now = time.monotonic()
                for index, job in enumerate(self.pending):
                    if not self.rate_limiter.ready(job.payload["model"], now):
                        continue
                    lease = self.slots.try_acquire() if self.slots else True
                    if lease is None:
                        break
                    self.pending.pop(index)
                    self.active += 1
                    self.rate_limiter.admitted(now)
                    self.metrics["peak_active"] = max(self.metrics["peak_active"], self.active)
                    return job, lease
                self.cv.wait(0.05)

    def _run_attempt(self, job, lease):
        result = AttemptResult(record={
            "attempt": len(job.attempts) + 1,
            "queue_seconds": time.monotonic() - job.submitted,
        })
        if hasattr(self.slots, 'label'):
            result.record['key_id'] = self.slots.label(lease)
        job.state = "running"
        started = time.monotonic()
        dispatched = False
        try:
            # Fail closed if dispatch intent cannot be recorded durably.
            if self.journal:
                self.journal.save(job)
            dispatched = True
            if hasattr(self.slots, 'dispatch'):
                status, headers, data = self.slots.dispatch(lease, job.path, job.payload)
            else:
                status, headers, data = self.transport(job.path, job.payload)
            result.data = data
            result.record["http_status"] = status
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            result.record["usage"] = usage
            billed = any(isinstance(value, (int, float)) and value > 0 for value in usage.values())
            if status == 200 and isinstance(data, dict) and not data.get("error"):
                if job.counting and (not isinstance(data.get("input_tokens"), int) or data["input_tokens"] < 0):
                    raise ValueError("Invalid provider token count")
            elif status in [429, 529] and not billed and len(job.attempts) + 1 < self.max_attempts:
                result.retry = True
                result.record["rejected"] = True
                delay = max(self.rate_limiter.retry_after(headers), min(300, 2 ** len(job.attempts)))
                with self.cv:
                    if hasattr(self.slots, 'rejected'):
                        self.slots.rejected(lease, status, delay)
                    else:
                        self.rate_limiter.rejected(status, job.payload["model"], delay)
                        if status == 429 and self.slots:
                            self.slots.rate_limited(str(delay))
                result.error = f"HTTP {status}: rejected, retry scheduled"
            else:
                result.error = f"Provider HTTP {status}"
                result.unknown = not job.counting and (status not in [429, 529] or billed)
        except Exception as exc:
            # Count requests are replayable. Generation transport failures
            # leave charging ambiguous and must never be silently replayed.
            result.error = type(exc).__name__
            result.unknown = not job.counting and dispatched
            result.retry = dispatched and job.counting and len(job.attempts) + 1 < min(3, self.max_attempts)
            if result.retry:
                with self.cv:
                    if hasattr(self.slots, 'rejected'):
                        self.slots.rejected(lease,529,min(4,2 ** len(job.attempts)))
                    else:
                        self.rate_limiter.rejected(529, job.payload["model"], min(4, 2 ** len(job.attempts)))
        finally:
            result.record["provider_seconds"] = time.monotonic() - started
            if result.error:
                result.record["error"] = result.error
            job.attempts.append(result.record)
            if self.slots:
                try:
                    self.slots.release(lease)
                except Exception:
                    # Retain the lease conservatively, but do not discard a
                    # completed response or lose the consumer thread.
                    result.record["lease_release_error"] = True
        return result

    def _finish_attempt(self, job, result):
        with self.cv:
            self.active -= 1
            self.metrics["provider_requests"] += 1
            if result.retry and self.closed:
                result.retry = False
                result.error = "Dispatcher shutdown; rejected request not retried"
            if result.retry:
                job.state = "queued"
                self.pending.append(job)
            else:
                self.incomplete -= 1
                job.finished = time.monotonic()
                if result.unknown:
                    self.metrics["unknown"] += 1
                job.state = "unknown" if result.unknown else "failed" if result.error else "succeeded"
                if job.counting:
                    self.count_flights.pop(job.fingerprint, None)
                if result.error:
                    job.future.set_exception(DispatchError(result.error, unknown=result.unknown))
                else:
                    if job.counting:
                        self.counts[job.fingerprint] = result.data
                        if len(self.counts) > self.cache_size:
                            self.counts.popitem(last=False)
                    job.future.set_result(result.data)
                    self.metrics["completed"] += 1
                    self.rate_limiter.succeeded()
            if self.journal:
                try:
                    usage = result.data.get("usage") if isinstance(result.data, dict) else None
                    self.journal.save(job, usage)
                except sqlite3.Error:
                    self.metrics["journal_errors"] = self.metrics.get("journal_errors", 0) + 1
            self.cv.notify_all()

    def _consumer(self):
        while True:
            item = self._take()
            if item is None:
                return
            job, lease = item
            self._finish_attempt(job, self._run_attempt(job, lease))

    def snapshot(self) -> dict:
        with self.cv:
            value = dict(
                self.metrics, queued=len(self.pending), active=self.active,
                incomplete=self.incomplete, interval_seconds=self.rate_limiter.interval,
            )
            if hasattr(self.slots, 'snapshot'):
                value['keys']=self.slots.snapshot()
            return value

    def close(self):
        """Cancel unsent jobs and drain in-flight calls without replaying them."""
        with self.cv:
            if self.closed:
                return
            self.closed = True
            for job in self.pending:
                job.state = "cancelled"
                job.finished = time.monotonic()
                job.future.set_exception(DispatchError("Cancelled before provider dispatch"))
                self.incomplete -= 1
                if self.journal:
                    self.journal.save(job)
            self.pending.clear()
            self.count_flights.clear()
            self.cv.notify_all()
        for thread in self.threads:
            thread.join()
        if hasattr(self.transport, "close"):
            self.transport.close()
        if hasattr(self.slots, 'dispatch'):
            self.slots.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
