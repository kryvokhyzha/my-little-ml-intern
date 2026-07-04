import json

import pytest

from intern.traces import TraceRecord, TraceStore, to_prompt_completion, to_sft_messages


def _record(task_id="task-1", accepted=True, messages=None, **overrides):
    return TraceRecord(
        task_id=task_id,
        split="train",
        messages=messages
        if messages is not None
        else [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        accepted=accepted,
        **overrides,
    )


class FakeTokenizer:
    """Deterministic stand-in for a HF tokenizer: renders each message as <role>content</role>."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None):
        assert tokenize is False
        rendered = f"<tools:{len(tools)}>" if tools else ""
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}</{message['role']}>"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class RewritingTokenizer:
    """Broken template that rewrites history per call, so the prompt is never a prefix of the full render."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, tools=None):
        rendered = f"<n={len(messages)}>" + "".join(message["content"] for message in messages)
        if add_generation_prompt:
            rendered += "<gen>"
        return rendered


@pytest.fixture
def store(tmp_path):
    return TraceStore(tmp_path / "traces.jsonl")


def test_append_read_roundtrip_preserves_all_fields(store):
    record = _record(
        tools=[{"type": "function", "function": {"name": "bash"}}],
        model_id="org/model",
        gen_params={"temperature": 0.7},
        steps=[{"action": "ls", "observation": "README.md"}],
        terminal_status="submitted",
        verifier_output={"tests_passed": 3},
        judge_critique="clean rollout",
        reward_components={"tests": 1.0, "style": 0.5},
    )
    store.append(record)
    store.append(_record(task_id="task-2", accepted=False))

    records = store.read()
    assert records == [record, _record(task_id="task-2", accepted=False)]
    lines = store.path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["reward_components"] == {"tests": 1.0, "style": 0.5}


def test_read_missing_file_returns_empty(store):
    assert store.read() == []
    assert store.accepted() == []


def test_read_skips_malformed_lines(store):
    store.append(_record())
    with store.path.open("a") as stream:
        stream.write('{"task_id": "task-2", "spl')  # simulated kill mid-write
        stream.write("\n[1, 2, 3]\n")
        stream.write('{"task_id": "task-3"}\n')  # missing required fields
        stream.write('{"task_id": "task-4", "split": "train", "messages": [], "bogus_field": 1}\n')

    records = store.read()
    assert len(records) == 1
    assert records[0].task_id == "task-1"


def test_accepted_filters_to_accepted_records(store):
    store.append(_record(task_id="keep", accepted=True))
    store.append(_record(task_id="drop", accepted=False))

    accepted = store.accepted()
    assert [record.task_id for record in accepted] == ["keep"]


def test_to_sft_messages_filters_and_shapes(store):
    tools = [{"type": "function", "function": {"name": "bash"}}]
    records = [_record(task_id="a", tools=tools), _record(task_id="b", accepted=False)]

    rows = to_sft_messages(records)
    assert rows == [{"messages": records[0].messages, "tools": tools}]

    all_rows = to_sft_messages(records, only_accepted=False)
    assert len(all_rows) == 2
    assert all_rows[1]["tools"] is None


def test_to_prompt_completion_happy_path():
    record = _record(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "draft"},
            {"role": "user", "content": "more"},
            {"role": "assistant", "content": "final"},
        ]
    )

    pairs = to_prompt_completion([record], FakeTokenizer())
    assert pairs == [
        {
            "prompt": "<user>hi</user><assistant>draft</assistant><user>more</user><assistant>",
            "completion": "final</assistant>",
        }
    ]


def test_to_prompt_completion_passes_tools_to_template():
    record = _record(tools=[{"type": "function", "function": {"name": "bash"}}])

    pairs = to_prompt_completion([record], FakeTokenizer())
    assert pairs[0]["prompt"].startswith("<tools:1>")
    assert pairs[0]["completion"] == "hello</assistant>"


def test_to_prompt_completion_skips_unaccepted_and_non_assistant_final():
    records = [
        _record(task_id="unaccepted", accepted=False),
        _record(task_id="dangling-user", messages=[{"role": "user", "content": "hi"}]),
        _record(task_id="empty", messages=[]),
        _record(task_id="good"),
    ]

    pairs = to_prompt_completion(records, FakeTokenizer())
    assert len(pairs) == 1
    assert pairs[0]["completion"] == "hello</assistant>"


def test_to_prompt_completion_raises_on_prefix_mismatch():
    with pytest.raises(ValueError, match="chat template"):
        to_prompt_completion([_record()], RewritingTokenizer())
