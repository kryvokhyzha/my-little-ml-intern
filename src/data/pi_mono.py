"""Convert badlogicgames/pi-mono raw session traces into SFT prompt/completion examples.

Ports the visible-turn conversion from the reference train_sft.py but builds
``intern.traces.TraceRecord`` objects (one per assistant turn) and delegates the
chat-template render — including the startswith completion-loss check — to
``intern.traces.to_prompt_completion``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from intern.traces import TraceRecord, to_prompt_completion


KNOWN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "description": "Run a shell command in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run."},
                "cmd": {"type": "string", "description": "Shell command to run."},
                "timeout": {"type": "number", "description": "Optional timeout in milliseconds."},
            },
            "required": [],
        },
    },
    "read": {
        "description": "Read a file or image from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to read."},
                "file": {"type": "string", "description": "Path to read."},
                "offset": {"type": "number", "description": "Optional starting line."},
                "limit": {"type": "number", "description": "Optional line limit."},
                "start": {"type": "number", "description": "Optional starting line."},
                "end": {"type": "number", "description": "Optional ending line."},
            },
            "required": [],
        },
    },
    "edit": {
        "description": "Edit a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to edit."},
                "oldText": {"type": "string", "description": "Text to replace."},
                "newText": {"type": "string", "description": "Replacement text."},
                "edits": {"type": "array", "description": "Structured edits."},
                "patch": {"type": "string", "description": "Patch content."},
            },
            "required": [],
        },
    },
    "write": {
        "description": "Write content to a file in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write."},
                "content": {"type": "string", "description": "File content."},
            },
            "required": [],
        },
    },
    "grep": {
        "description": "Search text in files.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
                "literal": {"type": "boolean", "description": "Treat pattern literally."},
                "context": {"type": "number", "description": "Context lines."},
            },
            "required": [],
        },
    },
    "find": {
        "description": "Find files or text in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Pattern to find."},
                "path": {"type": "string", "description": "Path to search."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "ls": {
        "description": "List files in a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path."},
                "limit": {"type": "number", "description": "Optional result limit."},
            },
            "required": [],
        },
    },
    "todo": {
        "description": "Manage a lightweight task list.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "Task-list action."},
                "text": {"type": "string", "description": "Task text."},
                "id": {"type": "string", "description": "Task identifier."},
            },
            "required": [],
        },
    },
}


def _generic_tool_schema(name: str) -> dict[str, Any]:
    return {
        "description": f"Pi coding-agent tool named {name}.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    omitted = len(text) - max_chars
    return f"{text[:head]}\n\n[... omitted {omitted} chars ...]\n\n{text[-tail:]}"


def _extract_text(parts: Any, max_chars: int, *, include_reasoning: bool) -> str:
    if isinstance(parts, str):
        return _clip_text(parts.strip(), max_chars)
    if not isinstance(parts, list):
        return ""
    out: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type == "text":
            value = str(part.get("text") or "").strip()
            if value:
                out.append(value)
        elif part_type == "image":
            out.append("[image omitted]")
        elif part_type == "thinking" and include_reasoning:
            value = str(part.get("thinking") or "").strip()
            if value:
                out.append(value)
    return _clip_text("\n".join(out).strip(), max_chars)


def _convert_tool_call(part: dict[str, Any]) -> dict[str, Any] | None:
    name = part.get("name")
    if not name:
        return None
    arguments = part.get("arguments")
    if arguments is None:
        arguments = {}
    call_id = part.get("id")
    if not call_id:
        digest = hashlib.sha1(json.dumps(part, sort_keys=True, default=str).encode()).hexdigest()[:12]
        call_id = f"call_{digest}"
    return {
        "id": str(call_id),
        "type": "function",
        "function": {"name": str(name), "arguments": arguments},
    }


def _assistant_message(
    raw_message: dict[str, Any], *, include_reasoning: bool, clip_chars: int
) -> dict[str, Any] | None:
    parts = raw_message.get("content") or []
    text = _extract_text(parts, clip_chars, include_reasoning=include_reasoning)
    tool_calls: list[dict[str, Any]] = []
    if isinstance(parts, list):
        for part in parts:
            if isinstance(part, dict) and part.get("type") == "toolCall":
                call = _convert_tool_call(part)
                if call is not None:
                    tool_calls.append(call)
    if not text and not tool_calls:
        return None
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _event_to_message(event: dict[str, Any], *, include_reasoning: bool, clip_chars: int) -> dict[str, Any] | None:
    if event.get("type") != "message":
        return None
    raw_message = event.get("message") or {}
    role = raw_message.get("role")
    if role == "user":
        content = _extract_text(raw_message.get("content") or [], clip_chars, include_reasoning=include_reasoning)
        return {"role": "user", "content": content} if content else None
    if role == "assistant":
        return _assistant_message(raw_message, include_reasoning=include_reasoning, clip_chars=clip_chars)
    if role == "toolResult":
        content = _extract_text(raw_message.get("content") or [], clip_chars, include_reasoning=include_reasoning)
        return {
            "role": "tool",
            "tool_call_id": str(raw_message.get("toolCallId") or ""),
            "name": str(raw_message.get("toolName") or "unknown"),
            "content": content or "[empty tool result]",
        }
    return None


def resolve_tools(events: list[dict]) -> list[dict]:
    """Return the tool schemas whose names appear as toolCall parts in a session's events."""
    names: set[str] = set()
    for event in events:
        if event.get("type") != "message":
            continue
        raw_message = event.get("message") or {}
        for part in raw_message.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "toolCall" and part.get("name"):
                names.add(str(part["name"]))
    tools: list[dict] = []
    for name in sorted(names):
        schema = KNOWN_TOOL_SCHEMAS.get(name, _generic_tool_schema(name))
        tools.append({"type": "function", "function": {"name": name, **schema}})
    return tools


def _trim_context(messages: list[dict[str, Any]], max_context_messages: int) -> list[dict[str, Any]]:
    context = list(messages)
    if max_context_messages > 0 and len(context) > max_context_messages:
        tail = context[-max_context_messages:]
        if not any(message.get("role") == "user" for message in tail):
            last_user = max(
                (index for index, message in enumerate(context) if message.get("role") == "user"),
                default=-1,
            )
            if last_user >= 0:
                tail = [context[last_user]] + context[-max(1, max_context_messages - 1) :]
        context = tail
    return context


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line {} in {}", lineno, path)
    return events


def sessions_to_trace_records(
    raw_dir: Path,
    *,
    include_reasoning: bool = False,
    max_context_messages: int = 18,
    per_turn_clip_chars: int = 12000,
    limit_sessions: int | None = None,
) -> list[TraceRecord]:
    """Emit one TraceRecord per visible assistant turn that has a preceding user message.

    ``messages`` is the trimmed prior context plus that assistant turn, ``tools`` the
    session's resolved schemas, ``task_id`` the session file stem, ``split='train'``,
    ``accepted=True``.
    """
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("*.jsonl"))
    if limit_sessions is not None:
        files = files[:limit_sessions]

    records: list[TraceRecord] = []
    for path in files:
        events = _load_events(path)
        tools = resolve_tools(events)
        conversation: list[dict[str, Any]] = []
        for event in events:
            message = _event_to_message(event, include_reasoning=include_reasoning, clip_chars=per_turn_clip_chars)
            if message is None:
                continue
            if message["role"] == "assistant" and any(m.get("role") == "user" for m in conversation):
                context = _trim_context(conversation, max_context_messages)
                records.append(
                    TraceRecord(
                        task_id=path.stem,
                        split="train",
                        messages=context + [message],
                        tools=tools or None,
                        accepted=True,
                    )
                )
            conversation.append(message)
    return records


def records_to_examples(records: list[TraceRecord], tokenizer, *, max_length: int = 4096) -> list[dict]:
    """Render records to prompt/completion pairs and drop those over ``max_length`` tokens.

    Rendering (and the completion-loss prefix assert) is delegated to
    ``intern.traces.to_prompt_completion``; length is measured on ``prompt + completion``.
    A record whose rendering violates the prefix assert is SKIPPED with a warning — one
    template-edge trace (e.g. consecutive assistant turns) must not kill a bulk conversion.
    """
    examples: list[dict] = []
    mismatched = 0
    for record in records:
        try:
            pairs = to_prompt_completion([record], tokenizer, only_accepted=False)
        except ValueError as err:
            mismatched += 1
            logger.warning("Skipping trace {}: {}", record.task_id, str(err).split(":")[0])
            continue
        if not pairs:
            continue
        pair = pairs[0]
        encoded = tokenizer(pair["prompt"] + pair["completion"], add_special_tokens=False)
        if len(encoded["input_ids"]) > max_length:
            continue
        examples.append({"prompt": pair["prompt"], "completion": pair["completion"], "task_id": record.task_id})
    if mismatched:
        logger.warning("Skipped {} record(s) with chat-template prefix mismatches", mismatched)
    return examples


def split_examples(examples: list[dict], *, eval_size: int = 256, seed: int = 42):
    """Shuffle deterministically and return a datasets.DatasetDict with train/test splits."""
    import random

    from datasets import Dataset, DatasetDict

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n_eval = min(eval_size, max(1, len(shuffled) // 20))
    return DatasetDict(
        train=Dataset.from_list(shuffled[n_eval:]),
        test=Dataset.from_list(shuffled[:n_eval]),
    )
