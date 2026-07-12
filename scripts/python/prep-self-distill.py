"""Run-once self-distillation prep: rollouts -> verify -> traces -> dataset (and post-train eval)."""

import sys

import hydra
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import DictConfig


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)

from pathlib import Path

from data.self_distill import build_arithmetic_tasks, collect_rollouts, success_rate
from intern.traces import TraceStore, to_prompt_completion
from training.models import load_model, load_tokenizer


def _tasks(cfg: DictConfig) -> tuple[list, list]:
    tasks = build_arithmetic_tasks(int(cfg.n_tasks), seed=int(cfg.seed), eval_fraction=float(cfg.eval_fraction))
    train = [task for task in tasks if task["split"] == "train"]
    eval_tasks = [task for task in tasks if task["split"] == "eval"]
    if cfg.limit_tasks is not None:
        train, eval_tasks = train[: int(cfg.limit_tasks)], eval_tasks[: int(cfg.limit_tasks)]
    return train, eval_tasks


@hydra.main(version_base=None, config_path="../../configs", config_name="prep-self-distill")
def main(cfg: DictConfig) -> None:
    experiment_dir = Path(str(cfg.experiment_dir))
    train_tasks, eval_tasks = _tasks(cfg)
    tokenizer = load_tokenizer(cfg)

    if cfg.eval_model_path is not None:
        # Post-train evaluation: same tasks, same verifier, the fine-tuned checkpoint.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(cfg.eval_model_path), dtype="float32")
        rate = success_rate(model, tokenizer, eval_tasks, max_new_tokens=int(cfg.max_new_tokens))
        print(
            f"SELF_DISTILL_EVAL | model={cfg.eval_model_path} | eval_tasks={len(eval_tasks)} | success_rate={rate:.3f}"
        )
        return

    model = load_model(cfg)
    baseline = success_rate(model, tokenizer, eval_tasks, max_new_tokens=int(cfg.max_new_tokens))
    logger.info("Baseline success rate on {} held-out eval tasks: {:.3f}", len(eval_tasks), baseline)

    records = collect_rollouts(
        model,
        tokenizer,
        train_tasks,
        k=int(cfg.k_samples),
        max_new_tokens=int(cfg.max_new_tokens),
        temperature=float(cfg.gen_temperature),
    )
    accepted = [record for record in records if record.accepted]
    logger.info(
        "Collected {} traces ({} accepted, rate {:.3f})",
        len(records),
        len(accepted),
        len(accepted) / max(1, len(records)),
    )

    if cfg.dry_run:
        logger.info("dry run — not writing traces or dataset")
        return
    if not accepted:
        raise SystemExit(
            "no accepted rollouts — the model never solved a train task; self-distillation has nothing to train on"
        )

    store = TraceStore(experiment_dir / "traces" / "rollouts.jsonl")
    for record in records:  # keep rejects too — preference-data material later
        store.append(record)

    from datasets import Dataset, DatasetDict

    pairs = to_prompt_completion(records, tokenizer, only_accepted=True)
    gold_rows = [
        {
            "prompt": tokenizer.apply_chat_template(
                [{"role": "user", "content": task["question"]}], add_generation_prompt=True, tokenize=False
            ),
            "completion": str(task["answer"]),
        }
        for task in eval_tasks
    ]
    data_dir = experiment_dir / "data"
    DatasetDict(train=Dataset.from_list(pairs), test=Dataset.from_list(gold_rows)).save_to_disk(str(data_dir))
    logger.success(
        "Wrote {} train pairs + {} gold eval rows to {} | baseline eval success {:.3f}",
        len(pairs),
        len(gold_rows),
        data_dir,
        baseline,
    )
    print(f"SELF_DISTILL_PREP | traces={len(records)} | accepted={len(accepted)} | baseline_success={baseline:.3f}")


if __name__ == "__main__":
    main()
