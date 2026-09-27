"""Load a dataset split from a local path (dir/file) or a Hub id, and validate task column contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


# On-policy lanes that sample completions from a ``prompt`` column (conversational or standard).
_PROMPT_ONLY_TASKS = (
    "trl_grpo",
    "trl_rloo",
    "trl_sdpo",
    "trl_ssd",
    "trl_distill",
    "trl_async_grpo",
    "trl_async_distill",
)
_FILE_FORMATS = {".json": "json", ".jsonl": "json", ".csv": "csv", ".parquet": "parquet", ".txt": "text"}


def load_split(dataset: str, split: str, for_eval: bool = False, **load_kwargs: Any) -> Any:
    """Load one split; extra kwargs are forwarded to the underlying datasets loader.

    Dispatch: an existing directory -> ``load_from_disk``; an existing file ->
    ``load_dataset(<format by suffix>)``; anything else -> ``load_dataset`` with the
    string as a Hub id. For needs beyond this (streaming, interleaving, custom
    builders), point the data config's ``_target_`` at ``datasets.load_dataset``
    directly instead of extending this function.
    """
    from datasets import DatasetDict, load_dataset, load_from_disk

    path = Path(dataset)
    if path.is_dir():
        ds = load_from_disk(str(path), **load_kwargs)
        if isinstance(ds, DatasetDict):
            return ds[split]
        if for_eval:
            raise ValueError(
                f"{dataset} is a plain on-disk Dataset with no splits — eval would silently "
                "run on the training data; save a DatasetDict or use one file per split"
            )
        return ds
    if path.is_file():
        fmt = _FILE_FORMATS.get(path.suffix.lower(), "json")
        return load_dataset(fmt, data_files=str(path), split=split, **load_kwargs)
    return load_dataset(dataset, split=split, **load_kwargs)


def validate_columns(dataset: Any, task: str, split: str, text_field: str = "text") -> None:
    """Fail fast (before any GPU spend) when the columns cannot feed the TRL task.

    Contracts (TRL dataset formats): SFT accepts ``text``/``text_field``,
    ``prompt``+``completion``, or ``messages``; DPO requires ``chosen``+``rejected``
    (``prompt`` optional — the implicit-prompt preference format is valid); KTO requires
    ``prompt``+``completion``+``label``; GKD requires ``messages``; GOLD accepts ``messages``
    or ``prompt``+``completion``; SDFT requires ``prompt``+``privileged_context``; every
    other on-policy lane (GRPO, RLOO, SDPO, SSD, distill, and the async lanes) requires
    ``prompt``. Unknown tasks are not checked. Extra columns are always allowed — e.g.
    tool-calling SFT ships ``messages`` + ``tools``, and reward functions receive every
    extra column as a kwarg.

    Raises:
        ValueError: When the dataset lacks every accepted column set for the task.

    """
    columns = set(getattr(dataset, "column_names", None) or [])
    if task == "trl_sft":
        accepted = f"'{text_field}', 'prompt'+'completion', or 'messages'"
        ok = text_field in columns or {"prompt", "completion"} <= columns or "messages" in columns
    elif task == "trl_dpo":
        accepted = "'chosen'+'rejected' ('prompt' optional)"
        ok = {"chosen", "rejected"} <= columns
    elif task in _PROMPT_ONLY_TASKS:
        accepted = "'prompt' (the on-policy trainer samples completions from prompts)"
        ok = "prompt" in columns
    elif task == "trl_gkd":
        accepted = "'messages' (GKD's collator consumes conversational data)"
        ok = "messages" in columns
    elif task == "trl_gold":
        accepted = "'messages' or 'prompt'+'completion'"
        ok = "messages" in columns or {"prompt", "completion"} <= columns
    elif task == "trl_sdft":
        accepted = "'prompt'+'privileged_context' (the self-teacher sees the context, the student does not)"
        ok = {"prompt", "privileged_context"} <= columns
    elif task == "trl_kto":
        accepted = "'prompt'+'completion'+'label' (unpaired per-example desirable/undesirable feedback)"
        ok = {"prompt", "completion", "label"} <= columns
    else:
        return
    if not ok:
        raise ValueError(f"{task} {split} dataset needs {accepted}; got columns {sorted(columns)}")
