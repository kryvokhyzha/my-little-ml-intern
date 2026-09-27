"""GuessNumberEnv against TRL's environment_factory contract."""

import inspect

from training.envs.guess_number import GuessNumberEnv, build_dataset


def test_public_methods_are_exactly_the_tools():
    # TRL exposes every public bound method except reset/get_reward as a tool.
    env = GuessNumberEnv()
    tools = [name for name, _ in inspect.getmembers(env, inspect.ismethod) if not name.startswith("_")]
    assert sorted(tools) == ["get_reward", "guess", "reset"]


def test_tool_schema_renders():
    from transformers.utils import get_json_schema

    schema = get_json_schema(GuessNumberEnv().guess)
    assert schema["function"]["name"] == "guess"
    assert schema["function"]["parameters"]["properties"]["number"]["type"] == "integer"


def test_reset_consumes_dataset_columns_and_game_scores():
    env = GuessNumberEnv()
    row = build_dataset(n=1)[0]
    assert env.reset(**row) is None  # the prompt already carries the instruction
    assert env.guess(row["low"] - 1) == "higher"
    assert env.guess(row["secret"]) == "correct"
    assert env.get_reward() == 1.0  # 2 guesses <= the binary-search bound of 7


def test_unsolved_and_slow_solves():
    env = GuessNumberEnv()
    env.reset(secret=50, low=1, high=100)
    assert env.get_reward() == 0.0
    for number in range(1, 51):
        env.guess(number)
    assert env.get_reward() == 0.1  # floor for a solve far past the bound


def test_eval_split_differs_from_train():
    assert build_dataset(n=20, split="train")["secret"] != build_dataset(n=20, split="eval")["secret"]


def test_gemma4_chat_template_parses_tool_calls():
    # TRL's bundled gemma4.jinja is prefix-preserving and parses a Gemma 4 tool call.
    # This test builds a tiny local tokenizer, so it needs no Hub access.
    from pathlib import Path

    import trl
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from trl import chat_template_utils as ctu

    specials = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<|turn>",
        "<turn|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|tool_response>",
        "<tool_response|>",
        "<|channel>",
        "<channel|>",
        '<|"|>',
        "<|tool>",
        "<tool|>",
    ]
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    bpe = trainers.BpeTrainer(
        vocab_size=300, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(["call guess n 42 hello model user"] * 10, bpe)
    template = (Path(trl.__file__).parent / "chat_templates" / "gemma4.jinja").read_text()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<eos>", pad_token="<pad>", bos_token="<bos>", chat_template=template
    )

    assert ctu.supports_tool_calling(hf_tok)
    assert ctu.is_chat_template_prefix_preserving(hf_tok)
    ctu.add_response_schema(hf_tok)
    assert hf_tok.response_template is not None

    prompt = [{"role": "user", "content": "guess"}]
    prefix = hf_tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    ids = hf_tok("<|tool_call>call:guess{n:42}<tool_call|>", add_special_tokens=False)["input_ids"]
    parsed = ctu.parse_response(hf_tok, ids, prefix=prefix)

    assert parsed["tool_calls"][0]["function"]["name"] == "guess"
    assert parsed["tool_calls"][0]["function"]["arguments"] == {"n": 42}
