# Verinuddle — agent notes

ComfyUI custom-node pack: four Save/Load Latent nodes that route through one
shared safetensors codec, which auto-detects `comfy.nested_tensor.NestedTensor`
samples (e.g. video+audio AV latents) and round-trips them instead of crashing
the way core's `SaveLatent` does. See `docs/latent_io.md` for the on-disk
schema and interop details before touching `latent_io.py` — it documents *why*
each metadata field exists, which is easy to get wrong by guessing.

## Layout

- `__init__.py` — package entry point ComfyUI loads; re-exports `comfy_entrypoint`.
- `src/verinuddle/__init__.py` — `ComfyExtension` node list.
- `src/verinuddle/latent_io.py` — the codec (`build_latent_tensors_and_metadata`
  / `latent_from_tensors_and_metadata`) plus the four node classes.
- `tests/test_latent_io.py` — codec tests, imported as a standalone module so
  they don't require a ComfyUI install (see the file's own docstring).
- `docs/latent_io.md` — on-disk schema, nested-stream contract, interop matrix.

## Nodes (id → title)

- `verinuddle_SaveLatent` → **Save Latent** (`output/`, filename_prefix + counter)
- `verinuddle_LoadLatent` → **Load Latent** (flat combo over `input/`)
- `verinuddle_ExportLatent` → **Save Latent (Path)** (arbitrary path)
- `verinuddle_ImportLatent` → **Load Latent (Path)** (arbitrary path)

The folder-pair titles intentionally match core's built-in "Save Latent"/"Load
Latent" node titles — that's deliberate, not an oversight. Node ids stay
`verinuddle_`-prefixed and distinct from core's so nothing gets shadowed.

## Conventions

- Stream order in a nested tensor is a contract, not an implementation detail:
  index 0 is video, later indices are e.g. audio. Never reorder streams.
- Non-nested latents must keep producing files byte-identical to core's own
  `SaveLatent` output — don't add codec fields that change that path.
- Missing/corrupt nested-stream metadata should raise a clear `ValueError`/
  `RuntimeError`, not silently truncate or guess.
- No dependency beyond what ComfyUI itself already provides at runtime
  (`torch`, `safetensors`, `folder_paths`, `comfy_api`) — keep `pyproject.toml`
  dependencies empty.

## Working here

- Run tests: `pytest tests/` (or, without a project-wide pytest install,
  `PYTHONPATH=<repo>:<path-to-venv>/site-packages python3.12 -m pytest tests/`).
  Only `torch`/`safetensors` are needed — the tests load `latent_io.py` by file
  path to avoid pulling in `comfy_api`/`folder_paths`.
- Lint: `ruff check .` (config lives in `pyproject.toml`).
- Keep commits scoped to one behavioral change, matching the existing history
  (scaffold → codec → path nodes → folder nodes → tests → docs/CI).
