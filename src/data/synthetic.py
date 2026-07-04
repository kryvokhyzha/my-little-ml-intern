"""Deterministic synthetic corpora for smoke experiments — no network needed."""

from __future__ import annotations

import random
from pathlib import Path


_SUBJECTS = ("the cat", "a small dog", "the little bird", "an old robot", "the young fox", "a tiny mouse")
_VERBS = ("walked to", "looked at", "ran past", "sat near", "jumped over", "slept beside")
_PLACES = ("the river", "the old house", "a tall tree", "the quiet garden", "the wooden bridge", "the green hill")
_ENDINGS = (
    "and then it started to rain.",
    "while the sun was setting.",
    "and everyone was happy.",
    "before the night came.",
    "and nothing else happened.",
    "as the wind blew softly.",
)


def build_tiny_text_dataset(path: Path | str, rows: int = 256, seed: int = 42) -> Path:
    """Materialize a small deterministic text dataset to disk in `datasets` format."""
    from datasets import Dataset

    rng = random.Random(seed)
    texts = []
    for _ in range(rows):
        sentences = [
            f"{rng.choice(_SUBJECTS).capitalize()} {rng.choice(_VERBS)} {rng.choice(_PLACES)} {rng.choice(_ENDINGS)}"
            for _ in range(rng.randint(3, 5))
        ]
        texts.append(" ".join(sentences))
    out = Path(path)
    Dataset.from_dict({"text": texts}).save_to_disk(str(out))
    return out
