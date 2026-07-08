"""Run-once prep: download badlogicgames/pi-mono, convert to SFT prompt/completion, push a private HF dataset."""

import sys

import hydra
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)

import os

from data.pi_mono import records_to_examples, sessions_to_trace_records, split_examples


def _resolve_owner(cfg: DictConfig) -> str:
    owner = cfg.target_repo or os.environ.get("HF_USER")
    if not owner:
        from huggingface_hub import HfApi

        owner = HfApi().whoami().get("name")
    if not owner:
        raise RuntimeError(
            "cannot resolve the target HF account: set `target_repo`, export HF_USER, or log in "
            "(HF_TOKEN) so whoami() returns an owner"
        )
    return str(owner).rstrip("/")


@hydra.main(version_base=None, config_path="../../configs", config_name="prep-pi-mono-sft")
def main(cfg: DictConfig) -> None:
    repo_id = f"{_resolve_owner(cfg)}/pi-mono-sft"

    raw_dir = hydra.utils.instantiate(cfg.data.source)  # data.pi_mono.download_sessions(...)
    raw_files = sorted(raw_dir.glob("*.jsonl"))
    logger.info("Downloaded {} raw *.jsonl files to {}", len(raw_files), raw_dir)

    tokenizer = hydra.utils.instantiate(OmegaConf.to_container(cfg.model.tokenizer, resolve=True))

    records = sessions_to_trace_records(
        raw_dir, include_reasoning=cfg.include_reasoning, limit_sessions=cfg.limit_sessions
    )
    examples = records_to_examples(records, tokenizer, max_length=cfg.max_length)
    ds = split_examples(examples, eval_size=cfg.eval_size)

    logger.info(
        "raw_files={} records={} examples_kept={} train={} test={}",
        len(raw_files),
        len(records),
        len(examples),
        len(ds["train"]),
        len(ds["test"]),
    )

    if cfg.dry_run:
        logger.info("dry run -- not pushing (would push {} to {})", repo_id, "private" if cfg.private else "public")
        return

    ds.push_to_hub(repo_id, private=cfg.private)
    logger.success("Pushed dataset to {} (private={})", repo_id, cfg.private)
    logger.info(
        "Set the experiment data.path to {!r}. To reproduce:\n  uv run python scripts/python/prep-pi-mono-sft.py",
        repo_id,
    )


if __name__ == "__main__":
    main()
