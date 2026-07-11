"""TRL lane: SFT / DPO / GRPO trainers driven by Hydra config."""

from training.trl.run import run_dpo, run_grpo, run_sft


__all__ = ["run_sft", "run_dpo", "run_grpo"]
