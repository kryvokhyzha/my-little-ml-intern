import json

import pytest

from data.pi_mono import (
    KNOWN_TOOL_SCHEMAS,
    records_to_examples,
    resolve_tools,
    sessions_to_trace_records,
    split_examples,
)


def _message_event(role, content):
    return {"type": "message", "message": {"role": role, "content": content}}


def _tool_result_event(tool_call_id, tool_name, text):
    return {
        "type": "message",
        "message": {
            "role": "toolResult",
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "content": [{"type": "text", "text": text}],
        },
    }


def _session_events():
    return [
        _message_event("user", [{"type": "text", "text": "list the files"}]),
        _message_event(
            "assistant",
            [
                {"type": "thinking", "thinking": "I should call bash to list the directory"},
                {"type": "text", "text": "Sure, let me look."},
                {"type": "toolCall", "id": "call-1", "name": "bash", "arguments": {"command": "ls"}},
            ],
        ),
        _tool_result_event("call-1", "bash", "README.md\nsrc"),
        _message_event("assistant", [{"type": "text", "text": "There are two entries: README.md and src."}]),
        {"type": "sessionMeta", "note": "ignored non-message event"},
    ]


def _write_session(raw_dir, name, events):
    path = raw_dir / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return path


class FakeTokenizer:
    """Deterministic stand-in: apply_chat_template joins <role>content</role>; __call__ counts words."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None):
        assert tokenize is False
        rendered = f"<tools:{len(tools)}>" if tools else ""
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}</{message['role']}>"
            for call in message.get("tool_calls") or []:
                rendered += f"<call>{call['function']['name']}</call>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def test_sessions_to_trace_records_one_per_assistant_turn(tmp_path):
    _write_session(tmp_path, "sess-a", _session_events())

    records = sessions_to_trace_records(tmp_path)

    assert [r.task_id for r in records] == ["sess-a", "sess-a"]
    assert all(r.split == "train" and r.accepted for r in records)

    first, second = records
    assert [m["role"] for m in first.messages] == ["user", "assistant"]
    assert first.messages[-1]["tool_calls"][0]["function"]["name"] == "bash"
    assert first.messages[-1]["tool_calls"][0]["id"] == "call-1"

    assert [m["role"] for m in second.messages] == ["user", "assistant", "tool", "assistant"]
    tool_message = second.messages[2]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "bash",
        "content": "README.md\nsrc",
    }
    assert second.messages[-1]["content"] == "There are two entries: README.md and src."


def test_thinking_dropped_by_default_kept_when_requested(tmp_path):
    _write_session(tmp_path, "sess", _session_events())

    default_records = sessions_to_trace_records(tmp_path)
    assert "I should call bash" not in default_records[0].messages[-1]["content"]

    reasoning_records = sessions_to_trace_records(tmp_path, include_reasoning=True)
    assert "I should call bash to list the directory" in reasoning_records[0].messages[-1]["content"]


def test_resolve_tools_matches_session_tool_calls():
    events = _session_events() + [
        _message_event("assistant", [{"type": "toolCall", "id": "c2", "name": "mystery", "arguments": {}}])
    ]

    tools = resolve_tools(events)
    names = [t["function"]["name"] for t in tools]
    assert names == ["bash", "mystery"]

    bash_tool = next(t for t in tools if t["function"]["name"] == "bash")
    assert bash_tool["function"]["description"] == KNOWN_TOOL_SCHEMAS["bash"]["description"]
    mystery_tool = next(t for t in tools if t["function"]["name"] == "mystery")
    assert "mystery" in mystery_tool["function"]["description"]


def test_trace_records_carry_resolved_tools(tmp_path):
    _write_session(tmp_path, "sess", _session_events())
    records = sessions_to_trace_records(tmp_path)
    assert [t["function"]["name"] for t in records[0].tools] == ["bash"]


def test_image_parts_omitted(tmp_path):
    events = [
        _message_event("user", [{"type": "text", "text": "look at this"}, {"type": "image", "url": "x"}]),
        _message_event("assistant", [{"type": "text", "text": "ok"}]),
    ]
    _write_session(tmp_path, "img", events)

    records = sessions_to_trace_records(tmp_path)
    assert "[image omitted]" in records[0].messages[0]["content"]


def test_per_turn_char_clip_applied(tmp_path):
    long_text = "x" * 500
    events = [
        _message_event("user", [{"type": "text", "text": long_text}]),
        _message_event("assistant", [{"type": "text", "text": "done"}]),
    ]
    _write_session(tmp_path, "clip", events)

    records = sessions_to_trace_records(tmp_path, per_turn_clip_chars=100)
    user_content = records[0].messages[0]["content"]
    assert len(user_content) < 500
    assert "omitted" in user_content


def test_context_trimming_keeps_user_anchor(tmp_path):
    events = [_message_event("user", [{"type": "text", "text": "kickoff"}])]
    for i in range(6):
        events.append(
            _message_event("assistant", [{"type": "toolCall", "id": f"c{i}", "name": "bash", "arguments": {}}])
        )
        events.append(_tool_result_event(f"c{i}", "bash", f"out {i}"))
    events.append(_message_event("assistant", [{"type": "text", "text": "final answer"}]))
    _write_session(tmp_path, "long", events)

    records = sessions_to_trace_records(tmp_path, max_context_messages=4)
    last = records[-1]
    assert last.messages[-1]["content"] == "final answer"
    assert any(m.get("role") == "user" for m in last.messages)


def test_records_to_examples_shape_and_prefix(tmp_path):
    _write_session(tmp_path, "sess", _session_events())
    records = sessions_to_trace_records(tmp_path)
    tokenizer = FakeTokenizer()

    examples = records_to_examples(records, tokenizer, max_length=4096)

    assert len(examples) == 2
    for example, record in zip(examples, records):
        assert set(example) == {"prompt", "completion", "task_id"}
        assert example["task_id"] == "sess"
        full = tokenizer.apply_chat_template(record.messages, tools=record.tools)
        assert full.startswith(example["prompt"])
        assert example["prompt"] + example["completion"] == full


def test_records_to_examples_drops_overlength(tmp_path):
    _write_session(tmp_path, "sess", _session_events())
    records = sessions_to_trace_records(tmp_path)
    tokenizer = FakeTokenizer()

    kept = records_to_examples(records, tokenizer, max_length=1)
    assert kept == []

    all_kept = records_to_examples(records, tokenizer, max_length=4096)
    assert len(all_kept) == 2


def test_records_to_examples_raises_on_prefix_mismatch(tmp_path):
    _write_session(tmp_path, "sess", _session_events())
    records = sessions_to_trace_records(tmp_path)

    class RewritingTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None):
            rendered = f"<n={len(messages)}>" + "".join(str(m.get("content", "")) for m in messages)
            if add_generation_prompt:
                rendered += "<gen>"
            return rendered

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": text.split()}

    with pytest.raises(ValueError, match="chat template"):
        records_to_examples(records, RewritingTokenizer())


def test_split_examples_deterministic_datasetdict():
    from datasets import DatasetDict

    examples = [{"prompt": f"p{i}", "completion": f"c{i}", "task_id": f"t{i}"} for i in range(100)]

    splits = split_examples(examples, eval_size=256, seed=42)
    assert isinstance(splits, DatasetDict)
    assert set(splits) == {"train", "test"}
    assert len(splits["test"]) == 5  # min(256, max(1, 100 // 20))
    assert len(splits["train"]) == 95

    again = split_examples(examples, eval_size=256, seed=42)
    assert splits["test"]["task_id"] == again["test"]["task_id"]

    other = split_examples(examples, eval_size=256, seed=7)
    assert splits["test"]["task_id"] != other["test"]["task_id"]


def test_split_examples_eval_size_cap_and_floor():
    small = [{"prompt": "p", "completion": "c", "task_id": "t"} for _ in range(3)]
    splits = split_examples(small, eval_size=256, seed=42)
    assert len(splits["test"]) == 1
    assert len(splits["train"]) == 2
