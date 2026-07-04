# 002 — Recommended `.claude/settings.json`

<!-- The permission allowlist the template's adoption checklist calls for. Not created
automatically: an agent widening its own permissions is blocked by design, so a human
must add this file. -->

Create `.claude/settings.json` with:

```json
{
  "permissions": {
    "allow": [
      "Bash(uv run *)",
      "Bash(uv tree*)",
      "Bash(uv pip list*)",
      "Bash(make *)",
      "Bash(pre-commit run *)",
      "Bash(bash scripts/bash/gpu_probe.sh*)",
      "Bash(bash scripts/bash/notify.sh *)",
      "Bash(ls *)",
      "Bash(git status*)",
      "Bash(git diff*)",
      "Bash(git log*)"
    ]
  }
}
```

Rationale: everything routine (`uv run`, `make`, gates via `intern.py`, probes,
notifications, read-only git) runs without permission prompts; anything mutating
outside that surface still asks.
