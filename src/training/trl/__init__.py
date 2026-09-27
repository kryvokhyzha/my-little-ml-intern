"""TRL lane: SFT / preference / online-RL / distillation / self-distillation trainers driven by Hydra config."""

from training.trl.run import (
    run_async_distill,
    run_async_grpo,
    run_distill,
    run_dpo,
    run_gkd,
    run_gold,
    run_grpo,
    run_kto,
    run_rloo,
    run_sdft,
    run_sdpo,
    run_sft,
    run_ssd,
)


__all__ = [
    "run_sft",
    "run_dpo",
    "run_kto",
    "run_grpo",
    "run_rloo",
    "run_gkd",
    "run_distill",
    "run_gold",
    "run_sdft",
    "run_sdpo",
    "run_ssd",
    "run_async_grpo",
    "run_async_distill",
]
