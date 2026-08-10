# Latent I/O

Four nodes, one shared codec (`src/verinuddle/latent_io.py`):

| Title | id | Location semantics |
|---|---|---|
| Save Latent | `verinuddle_SaveLatent` | `output/`, `filename_prefix` + counter, like core |
| Load Latent | `verinuddle_LoadLatent` | flat combo over `input/`, like core |
| Save Latent (Path) | `verinuddle_SaveLatentPath` | arbitrary absolute path |
| Load Latent (Path) | `verinuddle_LoadLatentPath` | arbitrary absolute path |

All four route through `build_latent_tensors_and_metadata` / `latent_from_tensors_and_metadata`.
The folder pair is a faithful mirror of core's `SaveLatent`/`LoadLatent` (same
`filename_prefix`/counter scheme, same flat `input/` listing, same fingerprinting) —
the only behavioural difference from core is the codec underneath.

## Why this exists

Core's `SaveLatent` crashes on `comfy.nested_tensor.NestedTensor` samples (e.g. MiniMax
H3's video+audio AV latents) with `AttributeError: 'NestedTensor' object has no
attribute 'contiguous'`, because `.latent`'s on-disk schema stores exactly one tensor
under `latent_tensor`. This codec extends that schema to store each stream as its own
sibling tensor, so nested latents save and load correctly, while non-nested latents
still produce a file byte-identical to core's own output.

## On-disk schema

**Extension: `.latent`.** It carries no format meaning — a `.latent` file *is* a
safetensors file. The extension matters in exactly one place in the whole backend: the
Load Latent listing filter, which keeps multi-GB model checkpoints out of the dropdown.
So it's a discovery contract, not a container choice — anything meant to be loadable
from `input/` must end in `.latent`.

**Non-nested case:** byte-identical to core — `latent_tensor` + `latent_format_version_0`,
plus (verinuddle-only) `noise_mask`/`batch_index`/`type` preserved in metadata, which
core's `SaveLatent` drops.

**Nested case:**

- **Tensors:** `latent_tensor` = stream 0, then `latent_tensor_1`, `latent_tensor_2`, …
  in `unbind()` order. Same scheme for masks: `noise_mask`, `noise_mask_1`, …
- **Metadata** (safetensors metadata is `str -> str` only):
  - `latent_format_version_0: "1"` — as core, drives the legacy 0.18215 fallback.
  - `nested_streams: "2"` — authoritative stream count; presence is the trigger for
    nested reconstruction (key order from `load_file` is not guaranteed, so this isn't
    inferred by key-sniffing).
  - `nested_noise_mask_streams: "2"` — masks can be nested independently of samples.
  - `verigen_latent_schema: "1"` — schema version for this codec's nested extension.

Stream order is the contract: `VAEDecode` takes `unbind()[0]` (video), audio decoders
take `unbind()[-1]`. There is no reordering step anywhere in this codec.

Missing an expected stream index raises a clear `ValueError` rather than silently
building a short nested tensor.

## Interop matrix

| File written by | Read by core Load Latent | Read by Verinuddle Load |
|---|---|---|
| core Save | full | full |
| Verinuddle, non-nested | full | full |
| Verinuddle, nested | stream 0 only (video), no error | full |

## Notes

- `Save Latent`/`Load Latent` titles intentionally match core's built-in node titles
  (ComfyUI allows duplicate titles across different node ids), so they're easy to find
  by search; their node ids (`verinuddle_SaveLatent`/`verinuddle_LoadLatent`) are
  distinct so they don't shadow the built-ins.
- `Save Latent (Path)`/`Load Latent (Path)` append `.latent` to the path when it has no
  extension already (so exports stay loadable from `input/` if moved there); the Path
  loader stays extension-agnostic so it can read `.safetensors` exports too.
- `comfy.nested_tensor` import is guarded (`try`/`except ImportError`) so this module
  still loads on ComfyUI builds that predate nested-tensor support; loading a nested
  file on such a build raises instead of silently truncating to stream 0.
