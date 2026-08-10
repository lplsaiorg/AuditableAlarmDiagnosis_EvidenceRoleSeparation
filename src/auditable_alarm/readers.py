"""Auditable reader construction with offline and OpenAI-compatible backends."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence
from urllib import request
from uuid import uuid4

from .audit import validate_reader_schema
from .hashing import canonical_bytes, sha256
from .models import ContrastCard, EventEvidence, ReaderRecord


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class ReaderBackend(Protocol):
    model_id: str
    decoding: Mapping[str, Any]

    def generate(self, messages: Sequence[ChatMessage]) -> str: ...


@dataclass(frozen=True)
class RuleBasedReaderBackend:
    """Deterministic offline reader for tests and service-free deployments."""

    benign_threshold: float = 0.62
    fault_threshold: float = 0.90
    model_id: str = "rule-reader-v1"

    @property
    def decoding(self) -> Mapping[str, Any]:
        return {"temperature": 0, "backend": "deterministic-rule"}

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        payload = json.loads(messages[-1].content)
        evidence = payload["current_event_evidence"]
        score = float(evidence["d_score"])
        aliases = sorted(evidence["aliases"])
        if score >= self.fault_threshold:
            label, confidence = "FAULT_OR_ATTACK", min(0.99, 0.80 + score * 0.19)
            support = aliases
        elif score <= self.benign_threshold:
            label = "BENIGN_LOOKALIKE"
            confidence = min(0.95, 0.80 + (self.benign_threshold - score) * 0.20)
            support = aliases
        else:
            label, confidence, support = "INSUFFICIENT_EVIDENCE", 0.50, []
        parsed = {
            "primary_label": label,
            "confidence": confidence,
            "supporting_evidence": support,
            "contradictions": [],
            "action": "REVIEW",
            "explanation": "Decision based only on frozen current-event evidence; history is analogy-only.",
            "missing_information": [] if support else ["Score lies in the abstention interval."],
        }
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


class OpenAICompatibleReaderBackend:
    """Minimal dependency-free client for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        timeout: float = 60.0,
    ) -> None:
        self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.temperature = float(temperature)
        self.timeout = float(timeout)

    @property
    def decoding(self) -> Mapping[str, Any]:
        return {"temperature": self.temperature, "response_format": "json_object"}

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        if not self.api_key:
            raise ValueError("LLM API key is not configured")
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": item.role, "content": item.content} for item in messages],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        target = self.base_url
        if not target.endswith("/chat/completions"):
            target += "/chat/completions"
        req = request.Request(
            target,
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except Exception as error:
            raise RuntimeError(f"LLM request failed: {error}") from error
        return str(result["choices"][0]["message"]["content"])


class ChatModelReaderBackend:
    """Adapter for an in-process object exposing ``generate(messages)``."""

    def __init__(self, chat_model: Any, model_id: str, decoding: Mapping[str, Any] | None = None) -> None:
        self.chat_model = chat_model
        self.model_id = model_id
        self.decoding = dict(decoding or {"temperature": 0})

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        module = sys.modules.get(self.chat_model.__class__.__module__)
        message_type = getattr(module, "ChatMessage", ChatMessage) if module else ChatMessage
        compatible = [message_type(role=item.role, content=item.content) for item in messages]
        return str(self.chat_model.generate(compatible))


SYSTEM_PROMPT = """You are an industrial alarm evidence reader. Return one JSON object only.
Required keys: primary_label, confidence, supporting_evidence, contradictions, action.
primary_label must be FAULT_OR_ATTACK, BENIGN_LOOKALIKE, or INSUFFICIENT_EVIDENCE.
supporting_evidence may contain only aliases from current_event_evidence.aliases.
Historical outcomes are analogies and must never be cited as current-event evidence.
Use REVIEW, LOG, MONITOR, or NOTIFY only; never prescribe autonomous control."""


class AuditableReader:
    def __init__(self, backend: ReaderBackend, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.backend = backend
        self.system_prompt = system_prompt

    def read(self, evidence: EventEvidence, history: ContrastCard) -> ReaderRecord:
        if evidence.query_id != history.query_id:
            raise ValueError("evidence/history query binding mismatch")
        payload = {
            "current_event_evidence": json.loads(canonical_bytes(evidence)),
            "historical_analogies": json.loads(canonical_bytes(history)),
        }
        input_bytes = canonical_bytes(payload)
        messages = (
            ChatMessage("system", self.system_prompt),
            ChatMessage("user", input_bytes.decode("utf-8")),
        )
        raw_text = self.backend.generate(messages)
        raw_bytes = raw_text.encode("utf-8")
        parsed: Mapping[str, Any]
        try:
            stripped = raw_text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
                if stripped.lstrip().startswith("json"):
                    stripped = stripped.lstrip()[4:].lstrip()
            candidate = json.loads(stripped)
            parsed = candidate if isinstance(candidate, dict) else {}
        except (ValueError, IndexError):
            parsed = {}
        return ReaderRecord(
            record_id=str(uuid4()),
            query_id=evidence.query_id,
            input_bytes=input_bytes,
            raw_output_bytes=raw_bytes,
            parsed=parsed,
            schema_valid=validate_reader_schema(parsed),
            evidence_hash=evidence.content_hash,
            history_hash=sha256(history),
            prompt_hash=sha256(self.system_prompt),
            schema_hash=sha256({"schema": "reader-output-v1"}),
            model_id=self.backend.model_id,
            decoding=self.backend.decoding,
        )
