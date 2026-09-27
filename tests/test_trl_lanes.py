"""End-to-end CPU smoke of the TRL lanes against the installed TRL, on tiny random local models.

These tests catch TRL API drift on an upgrade: each lane composes its real configs/trainer/<lane>.yaml,
builds its real trainer, and trains one step. The models are random 2-layer Qwen3-style checkpoints
with a byte-level BPE tokenizer and TRL's own qwen3 chat template (so tool calling parses) — no Hub
access. The async lanes need vLLM + CUDA, so a fake trainer checks their kwargs plumbing instead.
"""

import json
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize_config_dir


ROOT = rootutils.find_root(__file__, indicator=".project-root")
_SPECIALS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory) -> dict[str, Path]:
    import trl
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from data.self_distill import build_arithmetic_tasks

    out = tmp_path_factory.mktemp("tiny")
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    bpe = trainers.BpeTrainer(
        vocab_size=400, special_tokens=_SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(["What is 12 + 7? The answer is 19. guess number higher lower correct"] * 20, bpe)
    template = (Path(trl.__file__).parent / "chat_templates" / "qwen3.jinja").read_text()
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", chat_template=template
    )
    paths = {}
    for name, layers in (("student", 2), ("teacher", 3)):
        config = Qwen3Config(
            vocab_size=len(hf_tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            eos_token_id=hf_tok.eos_token_id,
            pad_token_id=hf_tok.pad_token_id,
            tie_word_embeddings=True,
        )
        Qwen3ForCausalLM(config).save_pretrained(out / name)
        hf_tok.save_pretrained(out / name)
        paths[name] = out / name
    rows: dict[str, list] = {"train": [], "test": []}
    for task in build_arithmetic_tasks(60):
        messages = [
            {"role": "user", "content": task["question"]},
            {"role": "assistant", "content": str(task["answer"])},
        ]
        rows["train" if task["split"] == "train" else "test"].append({"messages": messages})
    DatasetDict({k: Dataset.from_list(v) for k, v in rows.items()}).save_to_disk(str(out / "messages"))
    paths["messages"] = out / "messages"
    pairs = [{"prompt": f"What is {i} + 1?", "chosen": str(i + 1), "rejected": str(i + 2)} for i in range(40)]
    DatasetDict(train=Dataset.from_list(pairs)).save_to_disk(str(out / "preference"))
    paths["preference"] = out / "preference"
    unpaired = [
        {"prompt": pair["prompt"], "completion": pair[key], "label": key == "chosen"}
        for pair in pairs
        for key in ("chosen", "rejected")
    ]
    DatasetDict(train=Dataset.from_list(unpaired)).save_to_disk(str(out / "unpaired"))
    paths["unpaired"] = out / "unpaired"
    return paths


def _teacher(path: Path) -> str:
    return (
        f"+model.teacher={{_target_:transformers.AutoModelForCausalLM.from_pretrained,_args_:[{path}],dtype:float32}}"
    )


def _compose(lane: str, experiment_dir: Path, overrides: list[str]):
    base = [
        "+model=smollm2_135m_it",
        f"+trainer={lane}",
        "+tracking=none",
        "+compute=local",
        "+budget=smoke",
        f"experiment_name=999-{lane}",
        f"experiment_dir={experiment_dir}",
        "smoke_test=true",
        "trainer.args.per_device_train_batch_size=4",
        "trainer.args.gradient_accumulation_steps=1",
        "++trainer.args.use_cpu=true",
    ]
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        return compose(config_name="main", overrides=base + overrides)


_REWARD = "trainer.reward_funcs=[data.self_distill:arithmetic_reward]"
_SHORT = "++trainer.args.max_completion_length=8"
# lane -> overrides beyond the base; teacher / messages placeholders are filled from the fixture.
_LANES = {
    "trl_sft": ["+data=tiny_synthetic", "data.path={messages}"],
    "trl_sft_lora": ["+data=tiny_synthetic", "data.path={messages}", "trainer.peft.r=4"],
    "trl_dpo": ["+data=tiny_synthetic", "data.path={preference}"],
    "trl_kto": ["+data=tiny_synthetic", "data.path={unpaired}"],
    "trl_gkd": ["+data=tiny_synthetic", "data.path={messages}", "{teacher}", "trainer.args.max_new_tokens=8"],
    "trl_gold": ["+data=tiny_synthetic", "data.path={messages}", "{teacher}", _SHORT],
    "trl_distill": ["+data=arithmetic_prompts", "{teacher}", _SHORT],
    "trl_sdft": ["+data=arithmetic_prompts", _SHORT],
    "trl_sdpo": ["+data=arithmetic_prompts", _REWARD, _SHORT, "trainer.args.num_generations=4"],  # batch 4 here
    "trl_ssd": ["+data=arithmetic_prompts", _SHORT],
    "trl_rloo": ["+data=arithmetic_prompts", _REWARD, _SHORT],
    "trl_grpo": ["+data=arithmetic_prompts", _REWARD, _SHORT],
    "trl_grpo_env": ["+data=guess_number", "++trainer.args.max_completion_length=32"],
}


@pytest.mark.parametrize("lane", list(_LANES))
def test_lane_smoke_trains_one_step(lane, tiny, tmp_path):
    import training.trl as lanes

    paths = {"messages": tiny["messages"], "preference": tiny["preference"], "unpaired": tiny["unpaired"]}
    overrides = [o.format(teacher=_teacher(tiny["teacher"]), **paths) for o in _LANES[lane]]
    overrides.append(f"model.main._args_=[{tiny['student']}]")
    cfg = _compose(lane, tmp_path, overrides)
    run = getattr(lanes, "run_" + str(cfg.trainer.kind).removeprefix("trl_"))

    summary = run(cfg)

    assert summary["steps"] == 1
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["event"] == "run_start" and records[0]["task"] == cfg.trainer.kind
    assert any(r.get("name") == "loss" for r in records), "the lane logged no loss metric"
    if lane == "trl_dpo":
        assert any(r.get("name") == "rewards/margins" for r in records), "DPO logged no reward margin"
    if lane == "trl_grpo_env":
        # The environment's get_reward() became a reward source named after the env class.
        assert any(r.get("name") == "rewards/GuessNumberEnv/mean" for r in records)


class _FakeAsyncTrainer:
    """Stands in for AsyncGRPOTrainer / AsyncDistillationTrainer (they need vLLM + CUDA)."""

    captured: dict = {}

    def __init__(self, **kwargs):
        import torch
        from transformers import TrainerState

        type(self).captured = kwargs
        self.model = torch.nn.Linear(2, 2)
        self.state = TrainerState()

    def train(self):
        self.state.global_step = 1
        self.state.log_history.append({"loss": 0.5})


@pytest.mark.parametrize(
    ("lane", "module", "cls"),
    [
        ("trl_async_grpo", "trl.experimental.async_grpo", "AsyncGRPOTrainer"),
        ("trl_async_distill", "trl.experimental.async_distillation", "AsyncDistillationTrainer"),
    ],
)
def test_async_lane_passes_model_id_and_no_eval(lane, module, cls, tiny, tmp_path, monkeypatch):
    import importlib

    import training.trl as lanes

    monkeypatch.setattr(importlib.import_module(module), cls, _FakeAsyncTrainer)
    overrides = ["+data=arithmetic_prompts", "data.eval=null", f"model.main._args_=[{tiny['student']}]"]
    if lane == "trl_async_grpo":
        overrides += [_REWARD, "trainer.environment_factory=training.envs.guess_number:GuessNumberEnv"]
    cfg = _compose(lane, tmp_path, overrides)

    summary = getattr(lanes, "run_" + lane.removeprefix("trl_"))(cfg)

    kwargs = _FakeAsyncTrainer.captured
    assert kwargs["model"] == str(tiny["student"])  # the async trainers load the model by name
    assert "eval_dataset" not in kwargs and "peft_config" not in kwargs
    assert summary["param_count"] == 6  # counted from trainer.model after construction
    if lane == "trl_async_grpo":
        assert len(kwargs["reward_funcs"]) == 1
        assert kwargs["environment_factory"].__name__ == "GuessNumberEnv"


def test_lora_init_follows_cfg_seed(tiny, tmp_path):
    """TRL wraps PEFT before transformers seeds; _build_trainer must seed first so adapters reproduce."""
    from trl import SFTTrainer

    from intern.metrics import MetricsLog
    from training.trl.run import _build_trainer

    def lora_a(seed):
        overrides = ["+data=tiny_synthetic", f"data.path={tiny['messages']}", "trainer.peft.r=4", f"seed={seed}"]
        cfg = _compose("trl_sft_lora", tmp_path / str(seed), overrides + [f"model.main._args_=[{tiny['student']}]"])
        trainer, *_ = _build_trainer(cfg, SFTTrainer, None, False, MetricsLog(tmp_path / "m.jsonl"), True, False)
        return next(p.detach().clone() for n, p in trainer.model.named_parameters() if "lora_A" in n)

    import torch

    assert torch.equal(lora_a(7), lora_a(7))
    assert not torch.equal(lora_a(7), lora_a(8))


def add_numbers(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum.

    """
    return a + b


def test_distill_lane_wires_tools(tiny, tmp_path):
    """trainer.tools reaches DistillationTrainer (on-policy agent distillation, TRL >= 1.11)."""
    import training.trl as lanes

    overrides = [
        "+data=arithmetic_prompts",
        _teacher(tiny["teacher"]),
        _SHORT,
        f"trainer.tools=[{__name__}:add_numbers]",
        f"model.main._args_=[{tiny['student']}]",
    ]
    summary = lanes.run_distill(_compose("trl_distill", tmp_path, overrides))

    assert summary["steps"] == 1
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert any(r.get("name") == "tools/call_frequency" for r in records), "DistillationTrainer ran without tools"
