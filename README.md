# Verinuddle

Custom Save/Load Latent nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Why

Core's `SaveLatent`/`LoadLatent` write a single `latent_tensor` safetensors
entry and crash on `comfy.nested_tensor.NestedTensor` samples (e.g. video+audio
AV latents such as MiniMax H3's). Verinuddle extends the on-disk schema to
round-trip nested latents correctly, while still writing files that are
byte-identical to core's own output for ordinary (non-nested) latents.

## Nodes

| Title | Location semantics |
|---|---|
| Load Latent | flat combo over `input/`, like core |
| Save Latent | `output/`, `filename_prefix` + counter, like core |
| Load Latent (Path) | arbitrary path |
| Save Latent (Path) | arbitrary path |

See [docs/latent_io.md](docs/latent_io.md) for the on-disk schema and interop
details.

## Install

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/verigen/verinuddle.git
```

## Develop

```bash
cd verinuddle
pip install -e .[dev]
pre-commit install
pytest tests/
```
