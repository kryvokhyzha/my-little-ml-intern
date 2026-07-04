"""Post-training generation sampling for the verify generation_sanity check."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SAMPLE_PROMPTS = ("Once upon a time", "The weather this morning", "In a small village")


def write_samples(
    model: Any, tokenizer: Any, experiment_dir: Path, prompts: tuple[str, ...] = SAMPLE_PROMPTS, seed: int = 42
) -> None:
    import torch
    from loguru import logger

    records = []
    model.eval()
    torch.manual_seed(seed)
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
        records.append({"prompt": prompt, "text": tokenizer.decode(output[0], skip_special_tokens=True)})
    path = experiment_dir / "logs" / "samples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    logger.info("Wrote {} generation samples to {}", len(records), path)
