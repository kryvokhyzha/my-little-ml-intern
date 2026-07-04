"""Agent trace store and converters to SFT training shapes (docs/001 contract).

Traces live in ``experiments/NNN-<slug>/traces/*.jsonl`` (gitignored).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TraceRecord:
    """One agent rollout: full message history plus the judgment that decides whether it trains."""

    task_id: str
    split: str  # train | eval
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    model_id: str | None = None
    gen_params: dict[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None  # [{action, observation}]
    terminal_status: str | None = None
    verifier_output: dict[str, Any] | None = None
    judge_critique: str | None = None
    reward_components: dict[str, float] | None = None
    accepted: bool = False


class TraceStore:
    """Append-only JSONL store of TraceRecords, one ``dataclasses.asdict`` object per line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: TraceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    def read(self) -> list[TraceRecord]:
        if not self.path.exists():
            return []
        records: list[TraceRecord] = []
        for lineno, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line {} in {}", lineno, self.path)
                continue
            if not isinstance(payload, dict):
                logger.warning("Skipping non-dict JSONL line {} in {}", lineno, self.path)
                continue
            try:
                records.append(TraceRecord(**payload))
            except TypeError:
                logger.warning("Skipping JSONL line {} in {}: fields do not match TraceRecord", lineno, self.path)
        return records

    def accepted(self) -> list[TraceRecord]:
        return [record for record in self.read() if record.accepted]


def to_sft_messages(records: Iterable[TraceRecord], only_accepted: bool = True) -> list[dict[str, Any]]:
    """Convert traces to the conversational SFT shape: one ``{"messages", "tools"}`` dict per trace."""
    return [
        {"messages": record.messages, "tools": record.tools}
        for record in records
        if record.accepted or not only_accepted
    ]


def to_prompt_completion(
    records: Iterable[TraceRecord], tokenizer: Any, only_accepted: bool = True
) -> list[dict[str, str]]:
    """Render each trace's final assistant turn into a ``{"prompt", "completion"}`` pair.

    ``prompt`` is the chat template over every message before the final assistant one (with
    ``add_generation_prompt=True``); ``completion`` is the remainder of the full rendering.
    Raises ValueError on a chat-template prefix mismatch (the full rendering not starting with
    the prompt) — that mismatch silently corrupts completion-loss boundaries.
    """
    pairs: list[dict[str, str]] = []
    for record in records:
        if only_accepted and not record.accepted:
            continue
        if not record.messages or record.messages[-1].get("role") != "assistant":
            logger.warning("Skipping trace {}: last message is not from the assistant", record.task_id)
            continue
        context = record.messages[:-1]
        kwargs: dict[str, Any] = {"tools": record.tools} if record.tools else {}
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False, **kwargs)
        full = tokenizer.apply_chat_template(record.messages, tokenize=False, **kwargs)
        if not full.startswith(prompt):
            raise ValueError(
                f"chat template prefix mismatch for trace {record.task_id!r}: the full conversation rendering "
                "does not start with the rendered prompt — completion-loss boundaries would be corrupted"
            )
        pairs.append({"prompt": prompt, "completion": full[len(prompt) :]})
    return pairs
