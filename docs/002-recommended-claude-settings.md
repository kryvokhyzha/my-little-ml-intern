# 002 — Recommended `.claude/settings.json`

<!-- The deny rules ship committed in .claude/settings.json (narrowing agent permissions
is safe for an agent to do). The allow list below is NOT added automatically: an agent
widening its own permissions is blocked by design, so a human must add it. -->

The repo commits `.claude/settings.json` with the **deny** rules — agents never
commit or push; the human always owns history:

```json
{
  "permissions": {
    "deny": [
      "Bash(git commit)",
      "Bash(git commit *)",
      "Bash(git push)",
      "Bash(git push *)"
    ]
  }
}
```

Optionally merge in the **allow** list below (human-added by design):

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

Rationale: agents prepare changes, the human owns history (deny on commit/push);
everything routine (`uv run`, `make`, gates via `intern.py`, probes,
notifications, read-only git) runs without permission prompts; anything mutating
outside that surface still asks.
