---
name: literature-recipe-research
description:
  Isolated-context literature research that returns a ranked table of training
  recipes with published results, verified datasets, and working reference
  implementations. Use whenever the user says "research how to train X", "find
  the best recipe for", "find the best approach for", "what does the literature
  say", "survey methods for", or "find papers on" — and proactively before
  planning any training experiment whose method, dataset, or hyperparameters are
  not already fixed. Do not answer recipe questions from memory; run this skill
  instead.
context: fork
agent: general-purpose
---

# literature-recipe-research

Mine the literature for training recipes, attribute every claim to a published
result, and return ONE deliverable: a ranked recipe table plus a short
recommendation. This skill runs as a forked subagent — raw paper text, search
dumps, and API responses stay in this context; only the deliverable goes back to
the main agent.

## Rules

- Research-before-clarify: never ask the user about anything you could look up.
- Headless posture: never hang — write best-guess defaults, fire
  `bash scripts/bash/notify.sh approval_required "<message>"`, proceed.
- Doom-loop guard: 3 identical tool calls with no new information → write
  `blocker.md` (in the experiment dir when one is named, else
  `docs/999-blocker.md`), fire
  `bash scripts/bash/notify.sh blocker "<message>"`, stop.
- No vibes: every finding must be attributed as
  `Dataset X + method Y + hyperparams Z -> score W on benchmark V`. "They used
  SFT" is not a finding; drop anything you cannot attribute.
- Hard cap: stop researching at 1500 words of output — depth over breadth,
  methodology sections over abstracts.
- Context discipline: fetch targeted sections, never whole PDFs into the report;
  the final report contains zero raw tool output.

## Workflow

1. **Frame the task.** Restate in one paragraph for yourself: task/domain,
   target model scale, benchmark(s) that define success, compute context. Probe
   compute once with `bash scripts/bash/gpu_probe.sh` (`key=value` output:
   `cuda=`, `mps=`, `gpu_count=`, `gpu_name=`, `vram_gb=`). When an experiment
   is named, also read `experiments/NNN-<slug>/task.md` and `budget.md`
   (`compute_cap_gpu_h`, `scale_ceiling_params`) — these define the feasibility
   column.

2. **Find 2-3 anchor papers.** Prefer the alphaXiv MCP tools when available:
   `discover_papers` to search, `get_paper_content` to read
   (`answer_pdf_queries` for targeted questions). Fallback: WebSearch
   (`<task> training arxiv`, `site:arxiv.org <task>`) plus WebFetch on
   `https://arxiv.org/abs/<id>` and `https://huggingface.co/papers`. Pick
   anchors that are landmark (highly cited) or recent SOTA — ideally one of
   each.

3. **Crawl citations DOWNSTREAM from the anchors** — who improved on this, not
   what it cites. Via `discover_papers` with the anchor's key terms and a date
   filter after its publication, or:

   ```
   curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations?fields=title,year,citationCount,externalIds&limit=50"
   ```

   Prioritize recent + highly cited. If a downstream paper reports clearly
   better results, crawl its citations too (one extra hop max — hard cap).

4. **Read methodology sections specifically** (typically sections 3-5: method,
   experiments, results — never just the abstract). Use `get_paper_content`,
   else WebFetch `https://arxiv.org/pdf/<id>` or
   `https://ar5iv.labs.arxiv.org/html/<id>`. Extract per paper: exact dataset(s)
   (name, size, filtering/preprocessing), training config (optimizer, lr,
   schedule, epochs, batch size, seq length), and the exact scores those choices
   produced.

5. **Attribute.** Convert notes to findings in the required
   `dataset + method + hyperparams -> score on benchmark` form. Discard the
   rest.

6. **Verify assets exist before recommending them.** Use the `hf` CLI if
   installed, else the Hub API:

   ```
   curl -s "https://huggingface.co/api/datasets/<org>/<name>" | head -c 300
   curl -s "https://huggingface.co/api/models?search=<keywords>&limit=5"
   curl -s "https://datasets-server.huggingface.co/rows?dataset=<org>%2F<name>&config=default&split=train&offset=0&length=3"
   ```

   Check the column format matches the method: SFT needs `messages`, `text`, or
   `prompt`/`completion`; DPO needs `prompt`/`chosen`/`rejected`; GRPO needs
   `prompt`. A dataset or base model you did not verify does not go in the
   table.

7. **Find at least one working reference implementation per top recipe.**

   ```
   gh search repos "<method> <task>" --limit 5
   gh search code "<trainer or loss class>" --language python --limit 5
   ```

   `gh` needs auth (`GH_TOKEN` takes precedence over `GITHUB_TOKEN`), and
   unauthenticated `gh search` refuses to run. `.env` is NOT auto-loaded into
   the agent shell — check `gh auth status` first, and if needed export the
   token from `.env` for the session. On missing auth, fall back to WebSearch
   for repos instead of failing the step.

   Fallback: WebSearch `github <method> training script`. Fetch the linked file
   to confirm it exists before citing it.

8. **Write the deliverable** (output contract below) and save it:

   - `experiments/NNN-<slug>/research.md` when an experiment is named (`NNN` or
     full `NNN-slug` both identify it);
   - otherwise a numbered `docs/` entry per `.claude/skills/new-doc/SKILL.md`
     (next free `NNN-` prefix, kebab-case slug, e.g.
     `docs/003-<task>-recipe-research.md`).

   Return the same content as your final report — nothing else.

## Output contract

500-1500 words total. Two parts, nothing else — no preamble, no crawl log.

Ranked recipe table (best first), columns exactly:

```
| rank | method | dataset | key hyperparams | published result | source | reference impl | feasibility on our compute |
```

- `published result` — exact score + benchmark ("71.2 on MMLU"), not "strong".
- `source` — arXiv id or URL, with year.
- `reference impl` — repo/file URL you fetched and confirmed.
- `feasibility on our compute` — fits / tight / no, judged against
  `gpu_probe.sh` output and budget caps, one phrase of justification.

Then a 3-5 sentence recommendation: which recipe to implement first and why,
which verified dataset to use (exact Hub path), and any gaps — preprocessing
needed, method adaptation, license concerns.

## Done conditions

- [ ] 2-3 anchor papers identified; at least one downstream citation crawl done
- [ ] Every table row attributed: dataset + method + hyperparams -> score on
      benchmark, with source
- [ ] Every recommended dataset/model verified on the Hub (API or `hf` CLI
      output seen)
- [ ] At least one confirmed reference implementation URL per top recipe
- [ ] Output is 500-1500 words: ranked table + 3-5 sentence recommendation,
      nothing else
- [ ] Saved to `experiments/NNN-<slug>/research.md` or a numbered `docs/` file
