# ASD-STE100 Simplified Technical English — the rules this repo applies

ASD-STE100 is the controlled-language standard for aerospace maintenance
documentation. This repo applies its writing rules to every instruction file,
because agents act on these files literally. This file lists the rules, the two
deviations, and before/after examples from this repo.

## Where the rules apply

- **Full STE**: `AGENTS.md`, `CLAUDE.md`, `.claude/skills/**`, subagent prompts,
  the steps in `run.md`, CLI help text, and error messages.
- **STE sentence rules + Strunk**: `docs/`, the analysis in `results.md`, and
  `postmortems/`. These files explain tradeoffs, so they may argue and qualify.
  Strunk wins where the two guides conflict.

## The two deviations

1. The repo does not use the STE dictionary of about 900 approved words. Any ML,
   Python, or tooling term is a Technical Name, and the repo allows it.
2. `docs/` may argue and qualify. Procedure rules still apply to each step.

## Word rules

- Use one word for one meaning. Do not use a synonym for variety.
  - This repo says **gate** for a blocking check, **lane** for a trainer or
    compute adapter, and **path** for one solution attempt in an experiment.
- Do not use the `-ing` form as a verb. Write "run the smoke test", not "running
  the smoke test". An `-ing` form is allowed inside a technical name
  (`logging_steps`, `gradient_checkpointing`).
- Do not make a noun cluster of more than three nouns. Break it with a
  preposition: "the check of the reward variance", not "reward variance check
  result status".
- Keep the articles and the demonstratives: "the budget gate", "this path".
- Use the simple tenses: present, past, and future. Prefer the present tense.
- Do not use slang, idioms, or metaphors. Write the literal fact.
- Do not use a phrasal verb when a single verb exists: "start", not "kick off";
  "stop", not "shut down"; "examine", not "look into".

## Sentence rules

- Keep a procedural sentence to 20 words or fewer.
- Keep a descriptive sentence to 25 words or fewer.
- Write one topic per sentence.
- Do not delete words (articles, verbs) to make a sentence shorter.
- Use a vertical list for a complex condition or for more than two items.
- Make the items of a list parallel: all commands, or all noun phrases.

## Procedure rules

- Write each instruction as a command (imperative). Start it with the verb.
- Write one instruction per sentence. Two actions share a sentence only when the
  operator does them at the same time.
- Put the condition before the instruction: "If the gate exits 1, stop the run."
- Put a warning before the step that it applies to, never after it.
- Keep a procedural paragraph to 6 sentences.
- Number the steps when their order matters.

## Descriptive rules

- Keep a paragraph to one topic.
- Start the paragraph with the topic sentence.
- Give the information in steps: the fact first, then the detail.
- Use the active voice. Name the actor: "the gate refuses the run", not "the run
  is refused".

## Before and after

| Before                                                                          | After                                                                        |
| ------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Results.md should not be written until verification has been successfully run.  | Do not write results.md until `intern.py verify` exits 0.                    |
| Running the smoke test first is highly recommended before kicking off training. | Run the smoke test first. Then start the training run.                       |
| The gate, check, and guard all block the run.                                   | The gate blocks the run. (One word for one meaning.)                         |
| If you want, and the budget allows it, and the smoke passed, launch path 2.     | Launch path 2 only when all of these are true: (vertical list of conditions) |
| This robust, seamless integration leverages TRL's cutting-edge async trainer.   | The lane runs `AsyncGRPOTrainer`. It needs a vLLM server.                    |

## Self-check before you save

1. Does each instruction start with a verb?
2. Does each condition come before its instruction?
3. Is each procedural sentence 20 words or fewer?
4. Does each term keep one meaning in the whole file?
5. Did you remove each word that does not change the meaning?
