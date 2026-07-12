"""Self-distillation (STaR/RFT) building blocks: verifiable tasks, rollouts, deterministic verification.

The loop the distill-traces skill prescribes, on the smallest honest task: the model
generates k answers per TRAIN task, a deterministic verifier accepts the correct ones,
and the model is fine-tuned on its OWN accepted outputs. Held-out EVAL tasks measure
success rate before/after — they never yield training traces (contamination guard).
"""

from __future__ import annotations

import random
import re
from typing import Any

from loguru import logger

from intern.traces import TraceRecord


_INT_RE = re.compile(r"-?\d+")
_PROMPT_TEMPLATE = "What is {a} + {b}? Reply with only the number."


def build_arithmetic_tasks(n_tasks: int = 300, seed: int = 42, eval_fraction: float = 0.2) -> list[dict[str, Any]]:
    """Deterministic 2-digit-addition task pool with the train/eval split assigned up front.

    Unique (a, b) pairs so no eval task duplicates a train task; split membership is a
    property of the task, decided BEFORE any collection (the distill-traces rule).
    """
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_tasks:
        pairs.add((rng.randint(10, 99), rng.randint(10, 99)))
    ordered = sorted(pairs)
    rng.shuffle(ordered)
    n_eval = max(1, int(n_tasks * eval_fraction))
    tasks = []
    for index, (a, b) in enumerate(ordered):
        tasks.append(
            {
                "task_id": f"add-{index:04d}",
                "split": "eval" if index < n_eval else "train",
                "question": _PROMPT_TEMPLATE.format(a=a, b=b),
                "answer": a + b,
            }
        )
    return tasks


def verify_completion(completion: str, answer: int) -> dict[str, Any]:
    """Deterministic verifier: the LAST integer in the completion must equal the answer."""
    matches = _INT_RE.findall(completion)
    parsed = int(matches[-1]) if matches else None
    return {"correct": parsed == answer, "parsed": parsed}


def generate_answers(
    model: Any, tokenizer: Any, question: str, k: int, max_new_tokens: int, temperature: float
) -> list[str]:
    """Sample k completions for one task question through the model's chat template."""
    import torch

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            num_return_sequences=k,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    return [tokenizer.decode(sequence[prompt_length:], skip_special_tokens=True) for sequence in outputs]


def collect_rollouts(
    model: Any,
    tokenizer: Any,
    tasks: list[dict[str, Any]],
    *,
    k: int = 4,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
) -> list[TraceRecord]:
    """Roll out k samples per TRAIN task and verify each; eval tasks are refused outright."""
    records: list[TraceRecord] = []
    train_tasks = [task for task in tasks if task["split"] == "train"]
    if len(train_tasks) != len(tasks):
        raise ValueError("collect_rollouts received eval-split tasks — eval tasks never yield training traces")
    for index, task in enumerate(train_tasks):
        for completion in generate_answers(model, tokenizer, task["question"], k, max_new_tokens, temperature):
            verdict = verify_completion(completion, task["answer"])
            records.append(
                TraceRecord(
                    task_id=task["task_id"],
                    split="train",
                    messages=[
                        {"role": "user", "content": task["question"]},
                        {"role": "assistant", "content": completion.strip()},
                    ],
                    gen_params={"k": k, "max_new_tokens": max_new_tokens, "temperature": temperature},
                    verifier_output=verdict,
                    accepted=bool(verdict["correct"]),
                )
            )
        if (index + 1) % 25 == 0:
            accepted = sum(record.accepted for record in records)
            logger.info(
                "rollouts: {}/{} tasks, {} accepted / {} traces", index + 1, len(train_tasks), accepted, len(records)
            )
    return records


def success_rate(model: Any, tokenizer: Any, tasks: list[dict[str, Any]], *, max_new_tokens: int = 32) -> float:
    """Greedy one-shot success rate over the given tasks (use the EVAL split for the report)."""
    if not tasks:
        raise ValueError("success_rate needs at least one task")
    correct = 0
    for task in tasks:
        completion = generate_answers(model, tokenizer, task["question"], 1, max_new_tokens, temperature=0.0)[0]
        correct += bool(verify_completion(completion, task["answer"])["correct"])
    return correct / len(tasks)
