---
name: new-doc
description:
  Create a new numbered document in docs/ following the NNN-kebab-case-title.md
  convention. Use when the user asks to start a new doc, plan, analysis, design
  note, or research summary in this repo, or says things like "write this up",
  "save this as a doc", "new docs entry".
---

# new-doc

Scaffold a new Markdown doc under `docs/`. Use the repo's numbering convention.

## Naming convention (from AGENTS.md)

- `NNN-kebab-case-title.md` — the main numbered series for substantive docs. Use
  the **next free 3-digit prefix**.
- `099-*.md` — exploratory / experimental notes outside the main sequence.
- `999-*.md` — temporary scratch docs that you delete or fold back later.
  **Gitignored**. Use this prefix when the user does not want to commit the
  draft yet.
- Unprefixed `kebab-case.md` — standalone references (rationales, workflow
  guides) that are not part of a sequence.

Default to the main numbered series unless the user clearly asks for one of the
other variants.

## Steps

1. **Resolve the title.** If `$ARGUMENTS` is not empty, treat it as the
   human-readable title. Otherwise ask the user a short question: "What's the
   doc about?" (one line, no menu).
2. **Pick the filename.**
   - For the main series: run
     `ls docs/ 2>/dev/null | grep -E '^[0-9]{3}-' | sort | tail -1`. This
     command returns the highest prefix. Increment that prefix by 1. Zero-pad
     the result to 3 digits. If `docs/` does not exist yet, create it. Then
     start at `001`.
   - For the `099-` and `999-` variants: use that prefix directly. Do not
     increment it. Multiple `099-`/`999-` files can coexist.
   - Slugify the title to kebab-case (lowercase, non-alnum → `-`, collapse
     repeats, trim).
3. **Create the file** at `docs/<prefix>-<slug>.md` with this skeleton:

   ```text
   # <Title>

   <!-- Short one-line summary of what this doc covers and why it exists. -->

   ## Context

   ## Notes
   ```

   Adjust the skeleton when the user has already described the doc's purpose
   (e.g. a plan, a comparison, a bug investigation). Pick section headings that
   fit. Do not fill the body with placeholder prose.

4. **Report back** with the path as a markdown link (e.g.
   `[docs/003-foo.md](docs/003-foo.md)`). Then ask the user whether to draft the
   content now.

## Notes

- Never overwrite an existing file. If the computed path collides, increment the
  prefix or append `-v2`.
- Do not create `docs/` content that belongs in `trash/` (draft PR descriptions,
  throwaway patches, scratch test scripts).
- Do not write to `docs/build/`. That path is for generated output only.
