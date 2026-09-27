"""Post-training generation sampling for the verify generation_sanity check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAMPLE_PROMPTS = ("Once upon a time", "The weather this morning", "In a small village")


def _render_prompt(prompt: Any, tokenizer: Any) -> str:
    """Render one ``prompt`` cell: a string passes through; a conversation goes through the chat template."""
    if isinstance(prompt, list):
        if tokenizer is None:
            return ""
        return tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
    return str(prompt)


def resolve_sample_prompts(
    configured: Any, eval_dataset: Any, train_dataset: Any, n: int = 3, tokenizer: Any = None
) -> tuple[list[str], bool]:
    """Pick generation-probe prompts: explicit config > held-out prompt / messages column > defaults.

    Returns ``(prompts, pre_rendered)``. Prompt/completion datasets carry prompts already
    rendered in the model's chat format, so ``pre_rendered`` tells the tokenizer not to add
    its own special tokens — a fixed raw string would probe the model off-distribution.
    A conversational ``prompt`` column (a list of messages, as the on-policy and RL lanes use)
    is rendered through ``tokenizer``'s chat template; without a tokenizer it is skipped.
    A ``messages`` column (with a tokenizer) is rendered up to its final assistant turn.
    """
    if configured:
        return [str(p) for p in configured], False
    for dataset in (eval_dataset, train_dataset):
        columns = list(getattr(dataset, "column_names", None) or []) if dataset is not None else []
        if "prompt" in columns:
            rendered = [_render_prompt(p, tokenizer) for p in dataset[: min(n, len(dataset))]["prompt"]]
        elif "messages" in columns and tokenizer is not None:
            # Conversational data (GKD, GOLD, messages SFT): probe with the conversation up to its
            # final assistant turn — a raw "Once upon a time" string is off-distribution for a chat model.
            rendered = [_render_prompt(_context(m), tokenizer) for m in dataset[: min(n, len(dataset))]["messages"]]
        else:
            continue
        prompts = [p for p in rendered if p.strip()]
        if prompts:
            return prompts, True
    return list(SAMPLE_PROMPTS), False


def _context(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the trailing assistant turn so the model writes it."""
    return messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages


def write_samples(
    model: Any,
    tokenizer: Any,
    experiment_dir: Path,
    prompts: list[str] | tuple[str, ...] = SAMPLE_PROMPTS,
    pre_rendered: bool = False,
    seed: int = 42,
    max_prompt_tokens: int = 1024,
) -> None:
    import gc

    import torch
    from loguru import logger

    # Generation runs right after training, so free residual training memory first; a long
    # probe prompt on a big model otherwise OOMs (the run's real output is already saved).
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Keep the end of a pre-rendered chat prompt (the generation cue); the start otherwise.
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left" if pre_rendered else "right"

    records = []
    model.eval()
    torch.manual_seed(seed)
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=not pre_rendered,
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(model.device)
        try:
            with torch.no_grad():
                # Sampling, not greedy: sanity samples must reflect the model's distribution;
                # greedy decode loops even on healthy models and trips the repetition check.
                output = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=True,
                    top_k=50,
                    temperature=0.8,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # `text` repeats the prompt; `completion` holds only the new tokens, so the gate can see an
            # empty or looping answer behind a long chat prompt.
            new_tokens = output[0][len(inputs["input_ids"][0]) :]
            records.append(
                {
                    "prompt": prompt,
                    "text": tokenizer.decode(output[0], skip_special_tokens=True),
                    "completion": tokenizer.decode(new_tokens, skip_special_tokens=True),
                }
            )
        except torch.cuda.OutOfMemoryError as exc:
            logger.warning("Sample generation OOM ({}); skipping this prompt", exc)
            torch.cuda.empty_cache()

    tokenizer.truncation_side = prev_side
    path = experiment_dir / "logs" / "samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    logger.info("Wrote {} generation samples to {}", len(records), path)
