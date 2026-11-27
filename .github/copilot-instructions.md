# Working agreement

## Environment

- Python: conda env `bumblebee` (spelled with two `e`s: `bumblebee`, not `bumblebe`).
  Interpreter: `~/miniconda3/envs/bumblebee/bin/python`.
- Run code from `src/` with `PYTHONPATH=.` — modules are imported by bare name (`from explainer import ...`).

## Never modify my code without permission

- Propose diffs; apply only after I say so.
- **ALWAYS** apply source changes with the `edit`/`create` tools (VS Code workspace edits),
  so every change shows up in the editor as a reviewable hunk (Keep/Undo). Non-negotiable.
- **NEVER** write source files through `bash` — no scripts, `sed`, `python - <<EOF`, `cp`,
  `mv`, `rsync`, `tee`, heredocs, or redirections. There are no exceptions. If a change
  cannot be expressed with `edit`/`create`, stop and ask.
- Never modify a file merely to verify behaviour; read-only commands (`grep`, `sed -n`,
  `git diff`) are fine.
- When you need to test a variant, work on a copy.

## Scratch work: `.scratch/` inside this repo, never `/tmp`

- Copy the project into `.scratch/` (git-ignored) so I can inspect what you ran.
- Symlink the read-only data instead of copying it:
  `ln -s ../DATASETS ../MODELS .scratch/`
- Empty `.scratch/` when the task is done.

## Diffs must come from my current sources

- Re-copy from `src/` immediately before generating a diff; never diff against a stale copy.

## Project notes

- `OPTIMAL_BATCH_SIZE` in `parameters.py` is temporary (`2`); `8`/`64` are the targets.
- `graphic.py` draws; `explainer.py` holds the `Explainer` base class and `explanations()`;
  subclasses live in `explainer_att.py` and `explainer_ig.py`.
