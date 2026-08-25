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

Search the literature for training recipes. Attribute every claim to a published
result. Return ONE deliverable: a ranked recipe table plus a short
recommendation. This skill runs as a forked subagent. Raw paper text, search
dumps, and API responses stay in this context. Only the deliverable goes back to
the main agent.

## Rules

- Research-before-clarify: never ask the user about anything you can look up.
- Headless posture: never stop to wait for an answer. Write best-guess defaults.
  Fire `bash scripts/bash/notify.sh approval_required "<message>"`. Then
  continue.
- Doom-loop guard: after 3 identical tool calls that return no new information,
  write `blocker.md`. Write it in the experiment dir when the request names an
  experiment, else write `docs/999-blocker.md`. Fire
  `bash scripts/bash/notify.sh blocker "<message>"`. Then stop.
- No unattributed claims: attribute every finding as
  `Dataset X + method Y + hyperparams Z -> score W on benchmark V`. "They used
  SFT" is not a finding. Drop anything you cannot attribute.
- Hard cap: stop the research at 1500 words of output. Prefer depth over
  breadth. Prefer methodology sections over abstracts.
- Context discipline: fetch targeted sections. Never put a whole PDF into the
  report. The final report contains zero raw tool output.

## Workflow

1. **Frame the task.** Restate the task for yourself in one paragraph: the
   task/domain, the target model scale, the benchmark(s) that define success,
   and the compute context. Probe the compute once with
   `bash scripts/bash/gpu_probe.sh` (`key=value` output: `cuda=`, `mps=`,
   `gpu_count=`, `gpu_name=`, `vram_gb=`). When the request names an experiment,
   also read `experiments/NNN-<slug>/task.md` and `budget.md`
   (`compute_cap_gpu_h`, `scale_ceiling_params`). These two files define the
   feasibility column.

2. **Find 2-3 anchor papers.** Prefer the alphaXiv MCP tools when they are
   available: `discover_papers` to search, `get_paper_content` to read
   (`answer_pdf_queries` for targeted questions). Otherwise use WebSearch
   (`<task> training arxiv`, `site:arxiv.org <task>`) plus WebFetch on
   `https://arxiv.org/abs/<id>` and `https://huggingface.co/papers`. Pick
   anchors that are landmark (highly cited) or recent SOTA. Pick one of each
   when you can.

3. **Crawl citations DOWNSTREAM from the anchors.** Find the papers that
   improved on an anchor, not the papers that the anchor cites. Use
   `discover_papers` with the anchor's key terms and a date filter after its
   publication, or run:

   ```
   curl -s "https://api.semanticscholar.org/graph/v1/paper/arXiv:<id>/citations?fields=title,year,citationCount,externalIds&limit=50"
   ```

   Prioritize the recent and highly cited papers. When a downstream paper
   reports clearly better results, crawl its citations too. Take one extra hop
   at most. This cap is hard.

4. **Read the methodology sections** — typically sections 3-5: method,
   experiments, and results. Never read only the abstract. Use
   `get_paper_content`, else WebFetch `https://arxiv.org/pdf/<id>` or
   `https://ar5iv.labs.arxiv.org/html/<id>`. Extract the exact dataset(s) per
   paper: name, size, filtering/preprocessing. Extract the training config:
   optimizer, lr, schedule, epochs, batch size, seq length. Extract the exact
   scores that those choices produced.

5. **Attribute.** Convert your notes to findings in the required
   `dataset + method + hyperparams -> score on benchmark` form. Discard the
   rest.

6. **Verify that an asset exists before you recommend it.** Use the `hf` CLI
   when it is installed. Otherwise use the Hub API:

   ```
   curl -s "https://huggingface.co/api/datasets/<org>/<name>" | head -c 300
   curl -s "https://huggingface.co/api/models?search=<keywords>&limit=5"
   curl -s "https://datasets-server.huggingface.co/rows?dataset=<org>%2F<name>&config=default&split=train&offset=0&length=3"
   ```

   Check that the column format matches the method. SFT needs `messages`,
   `text`, or `prompt`/`completion`. DPO needs `prompt`/`chosen`/`rejected`.
   GRPO needs `prompt`. Never put an unverified dataset or base model in the
   table.

7. **Find at least one working reference implementation per top recipe.**

   ```
   gh search repos "<method> <task>" --limit 5
   gh search code "<trainer or loss class>" --language python --limit 5
   ```

   `gh` needs auth (`GH_TOKEN` takes precedence over `GITHUB_TOKEN`).
   Unauthenticated `gh search` refuses to run. The agent shell does NOT
   auto-load `.env`. Check `gh auth status` first. Export the token from `.env`
   for the session when you must. When `gh` has no auth, use WebSearch for repos
   and continue.

   Otherwise use WebSearch `github <method> training script`. Fetch the linked
   file. Confirm that it exists before you cite it.

8. **Write the deliverable** (see the output contract below). Save it:

   - `experiments/NNN-<slug>/research.md` when the request names an experiment
     (`NNN` or the full `NNN-slug` both identify it);
   - otherwise a numbered `docs/` entry per `.claude/skills/new-doc/SKILL.md`
     (next free `NNN-` prefix, kebab-case slug, e.g.
     `docs/003-<task>-recipe-research.md`).

   Return the same content as your final report. Return nothing else.

## Output contract

Write 500-1500 words in total. Write two parts and nothing else. Do not add a
preamble. Do not add a crawl log.

Write the ranked recipe table with the best recipe first. Use exactly these
columns:

```
| rank | method | dataset | key hyperparams | published result | source | reference impl | feasibility on our compute |
```

- `published result` — give the exact score and the benchmark ("71.2 on MMLU"),
  never "strong".
- `source` — give the arXiv id or the URL, with the year.
- `reference impl` — give the repo/file URL that you fetched and confirmed.
- `feasibility on our compute` — fits / tight / no. Judge it against the
  `gpu_probe.sh` output and the budget caps. Add one phrase of justification.

Then write a 3-5 sentence recommendation. Name the recipe to implement first and
give the reason. Name the verified dataset to use, with the exact Hub path. Name
any gaps: preprocessing needed, method adaptation, license concerns.

## Done conditions

- [ ] You identified 2-3 anchor papers. You did at least one downstream citation
      crawl
- [ ] You attributed every table row: dataset + method + hyperparams -> score on
      benchmark, with the source
- [ ] You verified every recommended dataset/model on the Hub (you saw the API
      or `hf` CLI output)
- [ ] You confirmed at least one reference implementation URL per top recipe
- [ ] The output is 500-1500 words: the ranked table plus a 3-5 sentence
      recommendation, nothing else
- [ ] You saved it to `experiments/NNN-<slug>/research.md` or to a numbered
      `docs/` file
