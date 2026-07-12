"""TRL lane: SFT / DPO / GRPO trainers driven by Hydra config."""

from training.trl.run import run_dpo, run_gkd, run_grpo, run_kto, run_sft


__all__ = ["run_sft", "run_dpo", "run_grpo", "run_gkd", "run_kto"]
