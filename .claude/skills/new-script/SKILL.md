---
name: new-script
description:
  Scaffold a new runnable Python entrypoint under scripts/python/ together with
  a paired Hydra config under configs/. Use when the user asks to add a new
  script, experiment, CLI, runner, or "new entrypoint" in this repo.
---

# new-script

Create a paired Hydra entrypoint. The pair is a `scripts/python/<name>.py`
script and a `configs/<name>.yaml` config. The config composes from the existing
`configs/main.yaml` defaults.

## Steps

1. **Resolve the name.** Use `$ARGUMENTS` when the user provides it. Otherwise
   ask the user for a short name (one line). Slugify the name to `kebab-case`
   for the file stem. The script keeps its hyphens: `python <path>` runs
   `scripts/python/foo-bar.py`, and no module imports it.
2. **Pick the prefix.**
   - The user sometimes signals that the script is exploratory or not ready to
     commit ("draft", "scratch", "WIP", "temporary"). Use the `999-` prefix for
     that script. Git ignores `scripts/python/999-*.py` and `docs/999-*.md`. The
     script stays local until you promote it.
   - Otherwise, check the existing config style in `configs/`. If numbered
     configs exist (for example `001-…yaml`, `002-…yaml`), use the next free
     3-digit prefix. This matches the convention.
   - Otherwise use the plain name.
3. **Create the config** `configs/<prefix?>-<name>.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - main
     - _self_
   # Script-specific parameters go here.
   ```

   Add only the parameters that the user describes. Do not invent any fields.

4. **Create the script** `scripts/python/<prefix?>-<name>.py`:

   ```python
   """<one-line description>."""

   import hydra
   import rootutils
   from dotenv import find_dotenv, load_dotenv
   from loguru import logger
   from omegaconf import DictConfig


   rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
   load_dotenv(find_dotenv(), override=True)


   @hydra.main(version_base=None, config_path="../../configs", config_name="<config-stem>")
   def main(cfg: DictConfig) -> None:
       """Entry point."""
       logger.info("Starting <name> with config:\n{}", cfg)
       # TODO: implement


   if __name__ == "__main__":
       main()
   ```

   Replace `<config-stem>` with the config filename without `.yaml`. Keep the
   `rootutils.setup_root(...)` call. The script finds the project root through
   this call. Keep `load_dotenv(find_dotenv(), override=True)` directly after
   it. The values in `.env` then take precedence over the ambient shell
   environment. This matches the project's logger / HF / WANDB conventions.

5. **Report back.** Link both files, for example
   `[scripts/python/003-foo.py](scripts/python/003-foo.py)` and
   `[configs/003-foo.yaml](configs/003-foo.yaml)`. Then show the run command:

   ```
   uv run python scripts/python/<file>.py
   ```

## Notes

- The script lives in `scripts/python/`, not in `src/`. `pyproject.toml` exempts
  `scripts/python/*` from `E402`. For this reason `rootutils.setup_root` and
  `load_dotenv(...)` can sit between the imports and other top-level code.
- `[project].dependencies` declares `python-dotenv` and `omegaconf`. Therefore
  `from dotenv import ...` and `from omegaconf import DictConfig` resolve on a
  freshly synced env. Run `make uv_install_deps` when you see an import error.
- Use `loguru.logger`. The project configures it globally. Do not instantiate
  `logging.getLogger`.
- Use Hydra for the configuration. Call `main()` directly under
  `if __name__ == "__main__":`. Do **not** wrap a `@hydra.main`-decorated
  function in `fire.Fire(...)`. Hydra and Fire both parse `sys.argv`. Fire then
  consumes Hydra's `key=value` overrides as the function's own arguments. The
  entrypoint fails. Use `fire.Fire(...)` only for a multi-command CLI that has
  **no** `@hydra.main` decorator. Do not add `argparse`/`click`.
- If the script needs reusable logic, move that logic into `src/<package>/...`.
  Keep `scripts/python/<name>.py` thin.
- Do not create the file under `trash/`. That directory holds throwaway scripts
  only. A new committed entrypoint belongs in `scripts/python/`.
