# 005 — Security defaults

<!-- The security-defaults contract: what is private by default and which code enforces
it, token hygiene, connection security, and the publish-bundle scrub gate. Companion to
docs/001-architecture.md (the contract) — this page is the paranoid appendix. -->

Everything that leaves this repo — checkpoints, metrics, bundles — is private
until a human explicitly says otherwise. "Private by default" is only a fact
where code enforces it, so the table below names the enforcing code for every
surface; anything marked doc-enforced is a convention you must uphold by hand.

## Private by default

| surface                 | enforcing code                                                                                                                                                                                                                             | default          |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- |
| publish gate            | `src/intern/publish.py` — `publish_run(..., private=True)`; the CLI accepts only bool or exact `true`/`false` for `--private` (anything else exits 2), a resolved `false` logs a loud warning, and the bundle scrub runs before any upload | private          |
| trackio Space + metrics | adapters force privacy from `tracking.private` (`configs/tracking/trackio.yaml`, default `true`): when `tracking.space_id` is set they pass `hub_private_repo=True`, so the Space and its metrics dataset are created private              | private          |
| hf_jobs Trainer pushes  | `hub_private_repo=True` is mandatory in trainer args for any run that pushes from the job — **doc-enforced**: no repo code can intercept a hand-rolled `push_to_hub`                                                                       | private          |
| wandb projects          | none in this repo — project visibility inherits from the **entity-level** setting on wandb.ai                                                                                                                                              | verify in the UI |

The publish gate is the only surface that can flip to public, and only via an
explicit `--private false` that survives the strict parser. Fire-and-forget
truthiness (`--private 0`, `--private 1`, floats) is rejected with exit 2
precisely because fire coerces CLI values — a typo must never publish weights to
the world.

## Token hygiene

- Use **fine-grained** HF tokens, never the legacy account-wide write token.
  Grant write scope only to the token used for `intern.py publish` and the
  hf_jobs lane; day-to-day downloads need read scope at most.
- `GITHUB_TOKEN` exists for `gh search` in the literature-recipe-research skill
  (unauthenticated `gh search` refuses). Keep it to public-repo read scope;
  `GH_TOKEN` takes precedence when both are set.
- **Rotate on any exposure** — pasted into a chat, a log, a screenshot, a
  committed file. Rotation costs a minute; a leaked write token costs your
  namespace.
- `scripts/bash/notify.sh` sources tokens from `.env` inside the script —
  callers pass only the event and message, so tokens never appear in shell
  history, agent transcripts, or the caller's command line.
- Tokens never live in configs: Hydra configs are committed files. If a config
  ever needs a secret, reference the environment via `${oc.env:VAR}` — never a
  literal value. Tokens never go to logs either; on a scrub hit the gate logs
  the pattern _name_, not the matched string, and so should you.

## Connection security

- **ssh lane** (`configs/compute/ssh.yaml`): key-only auth. Set the lane's
  `port`, `identity_file`, and `ssh_opts` keys instead of ad-hoc flags, and
  never enable password auth on a training box.
- Use **non-root** remote users; a trainer needs `uv` and a GPU, not uid 0.
- GCP/AWS boxes: restrict ssh with an **IP allowlist / security group** scoped
  to your address. Port 22 open to 0.0.0.0/0 is an invitation, not a lane.
- vast.ai: instances offer direct ssh or a proxied ssh endpoint — prefer direct
  with your own key when the offer supports it; either way the instance is just
  the ssh lane and all rules above apply.
- Dashboards (trackio, TensorBoard) on remote boxes are reached through
  `ssh -L <port>:localhost:<port>` port-forwarding — never bind a dashboard to
  `0.0.0.0` on a rented machine.

## What the publish bundle contains — and the scrub gate

`intern.py publish` uploads the newest `ckpts/` model dir plus a reproducibility
bundle: `task.md`, `plan.md`, `budget.md`, `ledger.md`, `verify.md`,
`results.md`, `logs/samples.jsonl`, and the experiment's
`configs/NNN-<slug>.yaml`, all under `bundle/` in the target repo.

Before anything is uploaded, every staged **text** file is scanned for:

- token-shaped strings — HF (`hf_…`), OpenAI (`sk-…`), Slack (`xox[abp]-…`), AWS
  (`AKIA…`), GitHub (`ghp_…`);
- home-directory absolute paths (`/Users/<name>/…`, `/home/<name>/…`) that leak
  usernames and machine layout.

On a hit the gate logs the file and the pattern name (never the matched value)
and exits 2 — nothing is uploaded. `INTERN_SKIP_BUNDLE_SCRUB=1` bypasses the
scan; use it only for a confirmed false positive, after reading the flagged file
yourself.

The scrub matches token _shapes_, not meaning: `logs/samples.jsonl` holds raw
model generations, and models regurgitate training data. If the dataset was
private or sensitive, eyeball the samples before publishing — no regex will
recognize your data as yours.

## Quick audit checklist

```bash
git check-ignore .env                          # prints ".env" — the secrets file is ignored
git grep -nE 'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|xox[abp]-'
                                               # empty output — no token-shaped strings tracked
uv run hf auth whoami                          # the expected account, the expected token
gh auth status                                 # GitHub token present, scopes minimal
grep -rn 'oc.env' configs/                     # the only legal secret path in configs — review every hit
ssh -G <host> | grep -iE 'identityfile|^port'  # ssh lane resolves to the intended key and port
lsof -nP -iTCP -sTCP:LISTEN | grep -v 127.0.0.1  # nothing listening beyond localhost
```

Any surprise in the output is a finding: fix it, rotate what leaked, and only
then go back to training.
