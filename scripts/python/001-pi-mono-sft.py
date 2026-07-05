"""SFT google/gemma-4-E2B-it on pi-mono agent traces via QLoRA (experiment 001)."""

import sys

import hydra
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import DictConfig


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="001-pi-mono-sft")
def main(cfg: DictConfig) -> None:
    logger.info("Starting 001-pi-mono-sft with config:\n{}", cfg)
    kind = cfg.trainer.kind
    if kind == "trl_sft":
        from training.trl import run_sft

        run_sft(cfg)
    else:
        raise ValueError(f"unknown trainer.kind: {kind}")


if __name__ == "__main__":
    main()
