---
name: new-script
description:
  Scaffold a new runnable Python entrypoint under scripts/python/ together with
  a paired Hydra config under configs/. Use when the user asks to add a new
  script, experiment, CLI, runner, or "new entrypoint" in this repo.
---

# new-script

Create a paired Hydra entrypoint: a `scripts/python/<name>.py` script and a
`configs/<name>.yaml` config that composes from the existing `configs/main.yaml`
defaults.

## Steps

1. **Resolve the name.** Use `$ARGUMENTS` if provided, otherwise ask the user
   for a short name (one line). Slugify to `kebab-case` for the file stem; the
   script is allowed to keep hyphens (`scripts/python/foo-bar.py` is fine since
   it's invoked via `python <path>`, not imported).
2. **Pick the prefix.**
   - If the user signals the script is exploratory / not ready to commit
     ("draft", "scratch", "WIP", "temporary"), use the `999-` prefix —
     `scripts/python/999-*.py` and `docs/999-*.md` are gitignored, so the file
     stays local until promoted.
   - Otherwise, check the existing config style in `configs/`: if numbered
     configs exist (e.g. `001-…yaml`, `002-…yaml`), use the next free 3-digit
     prefix to match the convention.
   - Otherwise use the plain name.
3. **Create the config** `configs/<prefix?>-<name>.yaml`:

   ```yaml
   # @package _global_
   defaults:
     - main
     - _self_
   # Script-specific parameters go here.
   ```

   Only add parameters the user has actually described — don't invent fields.

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
   `rootutils.setup_root(...)` call — it's how scripts find the project root.
   Keep `load_dotenv(find_dotenv(), override=True)` right after it so values in
   `.env` take precedence over the ambient shell environment (matches the
   project's logger / HF / WANDB conventions).

5. **Report back**: link both files (e.g.
   `[scripts/python/003-foo.py](scripts/python/003-foo.py)`,
   `[configs/003-foo.yaml](configs/003-foo.yaml)`) and show the run command:

   ```
   uv run python scripts/python/<file>.py
   ```

## Notes

- The script lives in `scripts/python/` (not `src/`) — `scripts/python/*` is
  exempt from `E402` in `pyproject.toml`, which is why `rootutils.setup_root`
  and `load_dotenv(...)` can sit between imports and other top-level code.
- `python-dotenv` and `omegaconf` are declared in `[project].dependencies`, so
  `from dotenv import ...` and `from omegaconf import DictConfig` resolve on a
  freshly synced env. If you ever see an import error, run
  `make uv_install_deps`.
- Use `loguru.logger` (already configured globally). Do not instantiate
  `logging.getLogger`.
- Use Hydra for configuration. Call `main()` directly under
  `if __name__ == "__main__":` — do **not** wrap a `@hydra.main`-decorated
  function in `fire.Fire(...)`. Hydra and Fire both parse `sys.argv`, so Fire
  swallows Hydra's `key=value` overrides (it treats them as the function's args)
  and the entrypoint breaks. Reach for `fire.Fire(...)` only for multi-command
  CLIs that have **no** `@hydra.main` decorator. Don't add `argparse`/`click`.
- If the script needs reusable logic, factor it into `src/<package>/...` and
  keep `scripts/python/<name>.py` thin.
- Don't create the file under `trash/` — that's reserved for throwaway scripts.
  New committed entrypoints belong in `scripts/python/`.
