"""Tiny SFT smoke experiment: exercises train -> metrics.jsonl -> verify on CPU/MPS."""

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

from training.toy_data import build_tiny_text_dataset
from training.trl_adapter import run_sft


@hydra.main(version_base=None, config_path="../../configs", config_name="001-tiny-sft-smoke")
def main(cfg: DictConfig) -> None:
    data_dir = Path(str(cfg.trainer.dataset))
    if not data_dir.exists():
        build_tiny_text_dataset(data_dir, seed=int(cfg.seed))
        logger.info("Materialized toy dataset at {}", data_dir)
    summary = run_sft(cfg)
    logger.info("Run summary: {}", summary)


if __name__ == "__main__":
    main()
