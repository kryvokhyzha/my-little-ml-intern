"""Guess-the-number: a minimal multi-turn tool environment for the GRPO environment lane.

The model gets a range and one tool, ``guess``. The environment answers "higher", "lower", or
"correct", and owns the reward through ``get_reward()``. The task is deterministic and needs no
service, so it proves the environment plumbing (tool parsing, multi-turn rollouts, env reward)
before a real environment (Harbor, OpenReward, an OpenEnv Space) replaces it.

TRL's contract (``GRPOTrainer(environment_factory=...)``): ``reset(**row)`` receives every
dataset column as a kwarg and may return text that TRL appends to the last user message; every
public method except ``reset``/``get_reward`` becomes a tool; ``get_reward()`` scores one rollout.
"""

from __future__ import annotations

import math
import random
from typing import Any


_PROMPT = (
    "I picked a secret integer between {low} and {high}. Find it with the `guess` tool. "
    "After each guess I answer 'higher', 'lower', or 'correct'. Use as few guesses as you can."
)


class GuessNumberEnv:
    """One rollout's game state. TRL builds one instance per concurrent rollout and reuses it via ``reset``."""

    def __init__(self) -> None:
        self.secret = 0
        self.low = 1
        self.high = 100
        self.guesses = 0
        self.solved = False

    def reset(self, secret: int = 0, low: int = 1, high: int = 100, **_: Any) -> None:
        self.secret, self.low, self.high = int(secret), int(low), int(high)
        self.guesses = 0
        self.solved = False

    def guess(self, number: int) -> str:
        """Guess the secret number.

        Args:
            number: The integer to guess.

        Returns:
            'higher' when the secret is larger, 'lower' when it is smaller, 'correct' when it matches.

        """
        self.guesses += 1
        number = int(number)
        if number == self.secret:
            self.solved = True
            return "correct"
        return "higher" if self.secret > number else "lower"

    def get_reward(self) -> float:
        """1.0 for a solve within the binary-search bound, minus 0.1 per extra guess; 0.0 when unsolved."""
        if not self.solved:
            return 0.0
        optimal = math.ceil(math.log2(self.high - self.low + 1))
        return max(0.1, 1.0 - 0.1 * max(0, self.guesses - optimal))


def build_dataset(n: int = 256, seed: int = 42, low: int = 1, high: int = 100, split: str = "train") -> Any:
    """Conversational prompts plus the ``secret``/``low``/``high`` columns that ``reset`` consumes.

    The eval split draws from a different seed stream, so its secrets and order differ from train.
    """
    from datasets import Dataset

    rng = random.Random(f"{seed}-{split}")
    prompt = [{"role": "user", "content": _PROMPT.format(low=low, high=high)}]
    return Dataset.from_list(
        [{"prompt": prompt, "secret": rng.randint(low, high), "low": low, "high": high} for _ in range(n)]
    )
