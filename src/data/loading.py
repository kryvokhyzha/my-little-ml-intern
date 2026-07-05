"""Load a dataset split from a local path (dir/file) or a Hub id, and validate task column contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
    (``prompt`` optional — the implicit-prompt preference format is valid); GRPO
    requires ``prompt``. Unknown tasks are not checked. Extra columns are always
    allowed — e.g. tool-calling SFT ships ``messages`` + ``tools``, and TRL forwards
    ``tools`` to the chat template.

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
    elif task == "trl_grpo":
        accepted = "'prompt' (GRPOTrainer samples completions from prompts)"
        ok = "prompt" in columns
    else:
        return
    if not ok:
        raise ValueError(f"{task} {split} dataset needs {accepted}; got columns {sorted(columns)}")
