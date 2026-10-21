# 012 — Google Colab compute

Colab runs the existing experiment entrypoints on a remote CPU or CUDA runtime.
The budget, smoke, ledger, journal, and verify rules remain the experiment
contract. Google's
[CLI announcement](https://developers.googleblog.com/introducing-the-google-colab-cli/)
describes the terminal workflow that this lane uses.

## Design

The [architecture](001-architecture.md#config-groups) assigns compute selection
to an agent-read parameter sheet. The
[Colab config](../configs/compute/colab.yaml) follows that contract. The
`train-llm` skill controls allocation, project transfer, the experiment gates,
artifact recovery, and teardown through the Google Colab CLI. This integration
needs no generic compute dispatcher.

The [execution helper](../src/intern/colab.py) handles one command on an
existing session. It accepts an explicit session name, state file, remote
directory, timeout, and optional log filename. It neither allocates nor stops
runtimes. The
[procedural reference](../.claude/skills/train-llm/references/colab.md) contains
the complete setup and experiment sequence.

### Remote exit codes

The upstream
[`exec_command`](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/commands/execution.py)
displays remote Python errors without a corresponding nonzero CLI exit. A
successful `colab exec` process therefore cannot prove that a remote gate
passed.

`intern.py colab-exec` executes the command as a remote subprocess. The helper
writes a status file with a unique execution ID and the subprocess exit code. It
downloads that file and validates both fields before it returns the remote exit
code. Missing, malformed, or mismatched status produces exit 2. A timeout
produces exit 124 when the remote wrapper confirms it. Treat an unconfirmed
outcome as a reason to inspect the runtime before a retry.

## Runtime choices

The config defaults to `gpu: T4` and `high_mem: false`. Set `compute.gpu=null`
for a CPU runtime. CPU runtimes support smoke tests; use CUDA for real training.
The official
[operator reference](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)
names these exact GPU selectors: `T4`, `L4`, `G4`, `H100`, and `A100`. Use an
exact selector. Unknown names can silently select A100.

Hardware availability depends on the account, subscription tier, quota, and
current capacity. A visible GPU choice does not guarantee allocation. The
[`new` command](https://github.com/googlecolab/google-colab-cli/blob/main/src/colab_cli/commands/session.py)
accepts `--high-mem`, subject to the account's entitlement. Some accelerators
have one machine shape and ignore that flag.

The supplied screenshot also shows TPU choices and a runtime-version selector.
The repo's trainers lack the XLA support that TPU training needs. This lane
therefore supports CPU and CUDA only. The CLI exposes no runtime-version
selector, so the config carries no equivalent field.

## Local setup

Install the optional `colab` dependency group on the local workstation:

```bash
uv sync --frozen --group colab --no-install-project
uv run --frozen --group colab colab version
```

The [PyPI release](https://pypi.org/project/google-colab-cli/0.7.4/) dates
version 0.7.4 to 2026-09-26. It meets the one-week rule on 2026-10-20. Version
0.7.1 appears in the upstream changelog but has no published PyPI files. The
group declares `google-colab-cli ~= 0.7.4`; `uv.lock` fixes the resolved
version. The existing seven-day resolver rule also covers its transitive
dependencies.

The CLI requires Python 3.12 or later. The group declares this requirement
without a change to the project's Python range. Use the project's Python 3.13
environment. Pass `--group colab` to local CLI and helper invocations. The
helper inherits the selected environment's PATH. Remote training omits this
group because only the workstation needs the CLI.

Complete the scoped ADC login from the procedural reference. The user must
complete the browser consent once. ADC authenticates the local CLI; `colab auth`
instead supplies credentials to notebook code. Keep the session state under
`trash/colab/sessions.json`, outside committed artifacts.

## Environment and transfer

Create a Python 3.13 uv environment on the runtime. Use `uv sync --frozen` with
the transferred `uv.lock`. The notebook kernel's preinstalled packages do not
define the training environment. Execute all project Python commands through uv.
Check the runtime driver against the lockfile's CUDA 12.9 torch wheel.

Transfer `.project-root`, `README.md`, `src/`, `configs/`, `scripts/`,
`pyproject.toml`, `uv.lock`, and the experiment's control artifacts. The
`.project-root` marker lets `rootutils` resolve the project. Exclude `.env`,
`.git`, `.venv`, caches, and local Colab session state. Supply runtime
credentials separately.

## Experiment lifecycle and evidence

Probe the allocated hardware. Run the smoke test on that runtime before scale.
Apply the existing budget and verify gates. Record the exact commands in
`experiments/NNN-<slug>/run.md`.

Download the complete experiment directory before teardown, including
checkpoints, metrics, logs, and journal entries. Verify the recovered artifacts
locally. Record GPU-hours for the allocation, including idle time. Stop the
owned session after artifact recovery. Colab can reclaim its ephemeral runtime
before a command's configured timeout.

Local transport tests exercise real subprocesses through a simulated Colab CLI.

## Live validation — 2026-10-20

### One-step smoke

The
[live transcript](../experiments/000-tiny-sft-smoke/run.md#colab-t4-smoke--2026-10-20)
records scoped ADC authentication, T4 allocation, project transfer, frozen
setup, artifact recovery, and confirmed teardown. The runtime uses Python
3.13.15 and torch 2.13.0+cu129. A CUDA tensor operation confirms the Tesla T4
and CUDA 12.9.

The smoke command executes one optimizer step and prints `TRAIN_OK`, with loss
3.6047396659851074. The full verify gate exits 1 because smoke mode consumes
only 544 tokens and omits generation samples. The failed report remains intact;
this check establishes no model-quality result. A separate archive preserves the
historical CPU baseline.

The final Colab API responses report zero active assignments and a zero usage
rate. The test also exposes a generated-code SyntaxWarning in the execution
helper. A local-variable fix removes it, and a regression test covers it.

### Full 10-step run

The
[full transcript](../experiments/000-tiny-sft-smoke/run.md#full-colab-t4-run--2026-10-20)
records a fresh T4 smoke and the complete recipe. Path-8 completes 10 steps,
consumes 5528 tokens, and produces three generation samples. Its final-step loss
is 1.7793203592300415. Its run-average loss is 2.3757517337799072.

The remote and recovered local verify gates exit 0: four checks pass and seven
checks skip absent evaluation or other-lane metrics. Direct language review
passes all three samples against the synthetic corpus. Repeated motifs and
100-token truncation limit the quality claim. The recipe saves no checkpoint.

The [results](../experiments/000-tiny-sft-smoke/results.md) link the recovered
artifacts and the preserved CPU baseline. The allocation uses at most 0.146404
GPU-hours, including setup and idle time. Colab confirms zero active assignments
and a zero usage rate after teardown.
