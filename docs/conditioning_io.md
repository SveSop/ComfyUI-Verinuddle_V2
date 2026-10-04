# Conditioning I/O

Five nodes, one shared codec (`src/verinuddle/conditioning_io.py`):

| Title | id | Location semantics |
|---|---|---|
| Save Conditioning | `verinuddle_SaveConditioning` | `output/`, `filename_prefix` + counter |
| Load Conditioning | `verinuddle_LoadConditioning` | flat combo over `input/` |
| Save Conditioning (Path) | `verinuddle_SaveConditioningPath` | arbitrary absolute path |
| Load Conditioning (Path) | `verinuddle_LoadConditioningPath` | arbitrary absolute path |
| Backup Conditioning (Path) | `verinuddle_BackupConditioningPath` | arbitrary absolute path, Save/Load merged behind a `save` switch |

All five route through `build_conditioning_tensors_and_metadata` /
`conditioning_from_tensors_and_metadata`. This mirrors the `latent_io.py` node set
(see `docs/latent_io.md`) node-for-node, but the codec itself is different in kind, not
just in name — see below.

## Why this exists

Nodes like `MiniMax H3 Image to Video` run an expensive CLIP/VAE encode to produce a
`CONDITIONING` that's normally wired straight into `BasicGuider` → a sampler. These
nodes let that conditioning be written to disk once and re-used across many separate,
independently-queueable sampling runs, instead of re-running the encode every time.

## Why the codec is a generic recursive walk, not a fixed key list

`LATENT`'s on-disk schema (see `latent_io.py`) special-cases a small, fixed set of keys:
`samples`, `noise_mask`, `batch_index`, `type`. `CONDITIONING` doesn't have that luxury —
it's `list[[tensor, dict]]`, and the dict's extra keys are open-ended and vary by which
node produced them (`pooled_output`, `minimax_keyframes`, `minimax_refs`, `area`, ...).
`minimax_keyframes`/`minimax_refs` in particular are themselves lists of dicts mixing
tensors with plain ints and strings. Hard-coding a key list here would silently drop
data the moment a new node introduces a new key, so instead the codec walks the whole
conditioning value recursively:

- Every `torch.Tensor` leaf is pulled out into a flat list and replaced in place by a
  placeholder marker `{"__tensor__": i}`.
- `dict`/`list` containers recurse structurally.
- `tuple` is preserved via a `{"__tuple__": [...]}` marker, since JSON (used for the
  metadata skeleton) has no tuple type.
- `int`/`float`/`str`/`bool`/`None` pass through unchanged.
- Anything else (e.g. a ControlNet object, or a live VAE/model reference) raises a clear
  `TypeError` naming the offending key path. This is a deliberate scope limit, not
  silent pickling: conditioning as produced by core and MiniMax H3 never contains such
  values today.
- A leaf that is itself a `comfy.nested_tensor.NestedTensor` raises a `RuntimeError`
  instead of being mis-serialized. Nothing currently nests a `NestedTensor` inside
  `CONDITIONING` (AV latents carry `NestedTensor` samples, not conditioning), so this is
  out of scope rather than a gap.

## On-disk schema

**Extension: `.conditioning`.** Same role as `.latent` for `latent_io.py`: it carries no
format meaning by itself (the file is a safetensors file), but it's the discovery
contract the Load Conditioning listing filters on.

- **Tensors:** every extracted tensor becomes its own safetensors entry, named
  `tensor_0`, `tensor_1`, ... in extraction order (a pre-order walk of the conditioning
  value).
- **Metadata** (safetensors metadata is `str -> str` only):
  - `conditioning_format_version_0: "1"` — schema version marker.
  - `conditioning_schema` — a JSON string: the walked conditioning skeleton, with every
    tensor leaf replaced by its `{"__tensor__": i}` placeholder and every tuple by its
    `{"__tuple__": [...]}` placeholder. Restoring is the inverse walk: replace each
    placeholder with the matching `tensor_i` (or a rebuilt tuple), recurse through
    dict/list, pass scalars through unchanged.

Missing schema metadata raises a clear `ValueError` rather than guessing a structure.

## Notes

- `Save Conditioning`/`Load Conditioning` follow the same folder-pair convention as
  `Save Latent`/`Load Latent` (`output/` with `filename_prefix` + counter; flat `input/`
  combo), for consistency within this pack.
- `Save Conditioning (Path)`/`Load Conditioning (Path)` append `.conditioning` to the
  path when it has no extension already; the Path loader stays extension-agnostic.
- `Backup Conditioning (Path)` reuses the exact lazy-input / `ExecutionBlocker`
  `stop_here` mechanism documented in `docs/latent_io.md`'s "Backup Latent (Path)"
  sections — see that file for the full mechanics; the only difference here is that the
  data being deferred/cut off is a `CONDITIONING` value instead of a `LATENT`.
