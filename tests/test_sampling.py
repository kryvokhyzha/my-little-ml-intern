import json

from datasets import Dataset

from training.sampling import SAMPLE_PROMPTS, resolve_sample_prompts, write_samples


def _pc(prompts):
    return Dataset.from_dict({"prompt": prompts, "completion": ["c"] * len(prompts)})


def _text(rows):
    return Dataset.from_dict({"text": rows})


class TestResolveSamplePrompts:
    def test_configured_wins_and_is_not_pre_rendered(self):
        prompts, pre = resolve_sample_prompts(["a", "b"], _pc(["e1", "e2"]), None)
        assert prompts == ["a", "b"]
        assert pre is False

    def test_eval_prompt_column_used_and_pre_rendered(self):
        prompts, pre = resolve_sample_prompts(None, _pc(["e1", "e2", "e3", "e4"]), _pc(["t1"]))
        assert prompts == ["e1", "e2", "e3"]  # first n=3, eval preferred over train
        assert pre is True

    def test_conversational_prompts_render_through_chat_template(self):
        class Tok:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                assert add_generation_prompt and not tokenize
                return "<user>" + messages[-1]["content"] + "<assistant>"

        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q1"}]]})
        prompts, pre = resolve_sample_prompts(None, dataset, None, tokenizer=Tok())
        assert prompts == ["<user>q1<assistant>"]
        assert pre is True

    def test_messages_rows_probe_with_the_context_before_the_last_assistant_turn(self):
        class Tok:
            def apply_chat_template(self, messages, add_generation_prompt, tokenize):
                return "|".join(m["content"] for m in messages) + "|<assistant>"

        convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        prompts, pre = resolve_sample_prompts(None, Dataset.from_dict({"messages": [convo]}), None, tokenizer=Tok())
        assert prompts == ["hi|<assistant>"]
        assert pre is True

    def test_conversational_prompts_without_tokenizer_fall_back(self):
        dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q1"}]]})
        prompts, pre = resolve_sample_prompts(None, dataset, None)
        assert prompts == list(SAMPLE_PROMPTS)
        assert pre is False

    def test_falls_back_to_train_prompt_column(self):
        prompts, pre = resolve_sample_prompts(None, None, _pc(["t1", "t2"]))
        assert prompts == ["t1", "t2"]
        assert pre is True

    def test_raw_text_dataset_uses_defaults(self):
        prompts, pre = resolve_sample_prompts(None, _text(["some text"]), _text(["more text"]))
        assert prompts == list(SAMPLE_PROMPTS)
        assert pre is False

    def test_no_datasets_uses_defaults(self):
        prompts, pre = resolve_sample_prompts(None, None, None)
        assert prompts == list(SAMPLE_PROMPTS)
        assert pre is False

    def test_blank_prompts_skipped(self):
        prompts, pre = resolve_sample_prompts(None, _pc(["  ", "real"]), None)
        assert prompts == ["real"]
        assert pre is True


class _FakeEnc(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    pad_token_id = 0
    truncation_side = "right"

    def __init__(self):
        self.add_special_tokens_calls: list[bool] = []
        self.decoded: list[list[int]] = []

    def __call__(self, text, return_tensors=None, add_special_tokens=True, truncation=False, max_length=None):
        self.add_special_tokens_calls.append(add_special_tokens)
        return _FakeEnc(input_ids=[[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        self.decoded.append(list(ids))
        return "generated continuation"


class _FakeModel:
    device = "cpu"

    def eval(self):
        pass

    def generate(self, **kwargs):
        return [[1, 2, 3, 4, 5]]


def test_write_samples_pre_rendered_disables_special_tokens(tmp_path):
    tok = _FakeTokenizer()
    write_samples(_FakeModel(), tok, tmp_path, prompts=["p1", "p2"], pre_rendered=True)
    assert tok.add_special_tokens_calls == [False, False]
    rows = [json.loads(line) for line in (tmp_path / "logs" / "samples.jsonl").read_text().splitlines()]
    assert [r["prompt"] for r in rows] == ["p1", "p2"]
    assert all(r["text"] == "generated continuation" for r in rows)
    assert all(r["completion"] == "generated continuation" for r in rows)
    assert tok.decoded[1] == [4, 5]  # the completion holds only the tokens after the 3-token prompt


def test_write_samples_raw_prompts_add_special_tokens(tmp_path):
    tok = _FakeTokenizer()
    write_samples(_FakeModel(), tok, tmp_path, prompts=["once upon a time"], pre_rendered=False)
    assert tok.add_special_tokens_calls == [True]
