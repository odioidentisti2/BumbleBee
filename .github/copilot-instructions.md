# Working agreement

## Environment

- Python: conda env `bumblebee` (spelled with two `e`s: `bumblebee`, not `bumblebe`).
  Interpreter: `~/miniconda3/envs/bumblebee/bin/python`.
- Run code from `src/` with `PYTHONPATH=.` — modules are imported by bare name (`from explainer import ...`).

## Never modify my code without permission

- Propose diffs; apply only after I say so.
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
