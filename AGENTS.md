# Verinuddle — agent notes

ComfyUI custom-node pack: Save/Load nodes for `LATENT` and `CONDITIONING` that route
through per-type safetensors codecs. The latent codec auto-detects
`comfy.nested_tensor.NestedTensor` samples (e.g. video+audio AV latents) and
round-trips them instead of crashing the way core's `SaveLatent` does. The
conditioning codec generically walks and reassembles the open-ended
`list[[tensor, dict]]` structure (e.g. MiniMax H3's `minimax_keyframes`/
`minimax_refs`) instead of hard-coding a fixed key list. See `docs/latent_io.md`
and `docs/conditioning_io.md` for on-disk schema and interop details before
touching either `*_io.py` module — they document *why* each metadata field
exists, which is easy to get wrong by guessing.

## Layout

- `__init__.py` — package entry point ComfyUI loads; re-exports `comfy_entrypoint`.
- `src/verinuddle/__init__.py` — `ComfyExtension` node list.
- `src/verinuddle/latent_io.py` — the LATENT codec (`build_latent_tensors_and_metadata`
  / `latent_from_tensors_and_metadata`) plus its five node classes.
- `src/verinuddle/conditioning_io.py` — the CONDITIONING codec
  (`build_conditioning_tensors_and_metadata` / `conditioning_from_tensors_and_metadata`)
  plus its five node classes.
- `tests/test_latent_io.py`, `tests/test_conditioning_io.py` — codec tests, imported
  as standalone modules so they don't require a ComfyUI install (see each file's own
  docstring).
- `docs/latent_io.md`, `docs/conditioning_io.md` — on-disk schema, nested-stream/
  generic-walk contract, interop matrix.

## Nodes (class → id → title)

- `SaveLatent` → `verinuddle_SaveLatent` → **Save Latent** (`output/`, filename_prefix + counter)
- `LoadLatent` → `verinuddle_LoadLatent` → **Load Latent** (flat combo over `input/`)
- `SaveLatentPath` → `verinuddle_SaveLatentPath` → **Save Latent (Path)** (arbitrary path)
- `LoadLatentPath` → `verinuddle_LoadLatentPath` → **Load Latent (Path)** (arbitrary path)
- `BackupLatentPath` → `verinuddle_BackupLatentPath` → **Backup Latent (Path)** (Save/Load
  merged behind a `save` switch; `samples` is a lazy input so Load mode never executes
  the upstream graph that would have produced it; `stop_here` does the opposite --
  in Save mode it returns an `ExecutionBlocker` so nothing downstream of this node's
  output executes either, letting a workflow split into independently-queueable phases)

- `SaveConditioning` → `verinuddle_SaveConditioning` → **Save Conditioning** (`output/`,
  filename_prefix + counter)
- `LoadConditioning` → `verinuddle_LoadConditioning` → **Load Conditioning** (flat combo
  over `input/`)
- `SaveConditioningPath` → `verinuddle_SaveConditioningPath` → **Save Conditioning
  (Path)** (arbitrary path)
- `LoadConditioningPath` → `verinuddle_LoadConditioningPath` → **Load Conditioning
  (Path)** (arbitrary path)
- `BackupConditioningPath` → `verinuddle_BackupConditioningPath` → **Backup
  Conditioning (Path)** (same save/lazy/`stop_here` semantics as `BackupLatentPath`,
  for splitting an expensive conditioning-producing workflow, e.g. MiniMax H3 encode,
  from the sampling phase that consumes it)

The folder-pair titles intentionally match core's built-in "Save Latent"/"Load
Latent" node titles — that's deliberate, not an oversight. (Core has no built-in
Save/Load Conditioning nodes, so the conditioning folder pair has no core
counterpart to match; the titles just follow the latent pair's naming style.)

Class names are plain and unprefixed: Python class names are scoped to this
package's own module and never collide with another custom-node pack's
classes, so a `Verinuddle`/`Verigen` prefix on the class buys nothing. Node
ids are different — those are registered in one global namespace shared by
every installed custom-node pack, so they keep the `verinuddle_` prefix to
stay unique and distinct from core's own `SaveLatent`/`LoadLatent` ids (which
would otherwise be shadowed).

## Conventions

- Stream order in a nested tensor is a contract, not an implementation detail:
  index 0 is video, later indices are e.g. audio. Never reorder streams.
- Non-nested latents must keep producing files byte-identical to core's own
  `SaveLatent` output — don't add codec fields that change that path.
- Missing/corrupt nested-stream metadata should raise a clear `ValueError`/
  `RuntimeError`, not silently truncate or guess.
- `conditioning_io.py`'s recursive walk raises `TypeError` on any leaf that isn't a
  tensor or a JSON-safe scalar/list/dict/tuple (e.g. a ControlNet object), and
  `RuntimeError` on a `NestedTensor` leaf — conditioning as produced by core/MiniMax H3
  never contains either today, so these are deliberate scope limits, not gaps to
  silently paper over by guessing an encoding.
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
