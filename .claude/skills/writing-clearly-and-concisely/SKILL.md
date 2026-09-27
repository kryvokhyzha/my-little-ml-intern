---
name: writing-clearly-and-concisely
description:
  Write and edit the prose that people and agents read in this repo — AGENTS.md,
  SKILL.md files, docs/, experiment artifacts (task.md, plan.md, results.md,
  postmortems), commit messages, PR text, error messages, and docstrings.
  Applies ASD-STE100 Simplified Technical English to instruction files, Strunk's
  Elements of Style to rationale prose, and removes common AI writing patterns.
  Use whenever you draft, rewrite, review, or shorten such text, or when the
  user says "edit for clarity", "make this concise", "STE", or "Simplified
  Technical English".
---

# Writing clearly and concisely

Agents read the files in this repo and act on them literally. An ambiguous
sentence causes a wrong action. This skill gives the rules, the workflow, and
the references for each type of text.

## Step 1 — pick the rule set

| Text                                                                                                | Rule set                                         | Read                                             |
| --------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `AGENTS.md`, `CLAUDE.md`, `.claude/skills/**`, subagent prompts, `run.md` steps, CLI/error messages | Full ASD-STE100                                  | `references/asd-ste100.md`                       |
| `docs/`, the analysis in `results.md`, `postmortems/`, PR descriptions                              | STE sentence rules + Strunk                      | `references/strunk-principles-of-composition.md` |
| Commit messages                                                                                     | Imperative subject ≤ 72 chars; the body says why | this file                                        |
| Docstrings, code comments                                                                           | STE sentence rules; Google style                 | `src/helper/logging/__init__.py`                 |

Where STE and Strunk conflict in an instruction file, STE wins. Where they
conflict in `docs/`, Strunk wins.

## Step 2 — write the draft

1. Put the main fact or the main instruction first.
2. Name the actor of each action: "the gate refuses the run".
3. Write each instruction as a command. Start it with the verb.
4. Put each condition before its instruction: "If verify exits 1, stop".
5. Use a vertical list for a complex condition or for more than two items.
6. Use the repo terms with one meaning each: **gate**, **lane**, **path**.

## Step 3 — apply the core rules

STE rules (instruction files):

- Keep procedural sentences to 20 words and descriptive sentences to 25.
- Give one instruction per sentence. Keep a procedural paragraph to 6 sentences.
- Use the active voice and the simple tenses. Prefer the present tense.
- Keep the articles. Write "the budget gate", not "budget gate".
- Do not use an `-ing` form as a verb. Do not use slang, idioms, or metaphor.
- Do not make a noun cluster of more than three nouns.

Strunk rules (all prose; the most useful six):

- Rule 10 — use the active voice.
- Rule 11 — put statements in the positive form: "ignores", not "does not pay
  attention to".
- Rule 12 — use definite, specific, concrete language: "loss=9.8 at step 120",
  not "the loss looked bad".
- Rule 13 — omit needless words: "because", not "due to the fact that".
- Rule 16 — keep related words together.
- Rule 18 — put the emphatic word at the end of the sentence.

## Step 4 — remove AI writing patterns

LLM prose regresses to generic, inflated language. Delete these patterns:

- **Puffery**: pivotal, crucial, vital, key role, testament, enduring,
  landscape.
- **Promotional adjectives**: robust, seamless, cutting-edge, groundbreaking,
  powerful, comprehensive.
- **AI vocabulary**: delve, leverage, multifaceted, foster, realm, tapestry,
  showcase, underscore.
- **Empty `-ing` tails**: "…, ensuring reliability", "…, highlighting the
  importance of X". State the fact, or delete the tail.
- **Hedges and throat-clearing**: "It is important to note that", "may
  potentially", "In order to".
- **Format noise**: bold on every other phrase, emoji, a bullet list for content
  that is one sentence.
- **Vague claims**: "improved performance". Give the number, the metric, and the
  comparison.

Read `references/signs-of-ai-writing.md` when you review a long text for AI
patterns. It is the Wikipedia field guide, about 25,000 tokens. Read only the
section that you need.

## Step 5 — cut and check

1. Read each sentence. Delete each word that does not change the meaning.
2. Replace each vague word with the number, the file, or the command.
3. Examine each term. It must keep one meaning in the whole file.
4. Run the self-check at the end of `references/asd-ste100.md`.

## Examples

| Before                                                                                      | After                                                       |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| It is important to note that the verify gate might potentially fail if samples are missing. | The verify gate fails when `logs/samples.jsonl` is missing. |
| This commit implements functionality for ensuring the budget is properly handled.           | Deny the launch when the budget cap is spent                |
| The model showed significantly improved performance.                                        | Held-out task success rose from 0.867 to 0.950 (n=60).      |
| Running the smoke test is recommended.                                                      | Run the smoke test first.                                   |

## References — read X when Y

- `references/asd-ste100.md` — read before you write or edit an instruction
  file.
- `references/strunk-principles-of-composition.md` — read for paragraph
  structure, the active voice, and concision. Most edits need only this file.
- `references/strunk-rules-of-usage.md` — read for commas, possessives, and
  sentence structure.
- `references/strunk-matters-of-form.md` — read for headings, quotations, and
  numerals.
- `references/strunk-words-commonly-misused.md` — read when you are unsure of a
  word choice.
- `references/signs-of-ai-writing.md` — read when you review a long text for AI
  patterns.

## Limited context

When the context is small:

1. Write the draft with the rules in this file.
2. Start a subagent. Give it the draft and one reference file.
3. Tell the subagent to return the edited text only.

## Done-condition

- [ ] Each instruction starts with a verb, and each condition precedes it.
- [ ] No sentence in an instruction file exceeds the STE word limit.
- [ ] No AI-pattern word from Step 4 remains.
- [ ] Each claim carries a number, a path, or a command where one exists.

## Attribution

Adapted from `softaworks/agent-toolkit` `skills/writing-clearly-and-concisely`
(MIT; see `LICENSE` in this directory), which adapts
`joshuadavidthomas/agent-skills` and `obra/the-elements-of-style`. The Strunk
text (1918) is in the public domain. `references/signs-of-ai-writing.md` is from
Wikipedia and keeps its CC BY-SA 4.0 license. The STE rules and the repo
adaptations are local.
