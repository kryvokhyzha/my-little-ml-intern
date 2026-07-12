"""Off-policy distillation: SFT SmolLM2-135M-Instruct on smoltalk teacher completions (experiment 002)."""

import sys

import hydra
import rootutils
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from omegaconf import DictConfig


root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
sys.path.insert(0, str(root / "src"))
load_dotenv(find_dotenv(), override=True)


@hydra.main(version_base=None, config_path="../../configs", config_name="002-distill-off-policy")
def main(cfg: DictConfig) -> None:
    logger.info("Starting 002-distill-off-policy with config:\n{}", cfg)
    kind = cfg.trainer.kind
    if kind == "trl_sft":
        from training.trl import run_sft

        run_sft(cfg)
    else:
        raise ValueError(f"unknown trainer.kind: {kind}")


if __name__ == "__main__":
    main()
