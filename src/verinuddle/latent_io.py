"""Latent import/export nodes and the shared safetensors schema they use.

The on-disk schema mirrors core's own SaveLatent/LoadLatent (.latent) format
("latent_tensor" key + "latent_format_version_0" marker tensor, with the
legacy SD1.x rescale fallback when that marker is absent) so files are
interchangeable with ComfyUI's built-in Save Latent / Load Latent nodes.
It's extended in two ways core isn't:

- noise_mask/batch_index/type are preserved via safetensors metadata, which
  core's SaveLatent currently drops.
- comfy.nested_tensor.NestedTensor samples (e.g. MiniMax H3's video+audio AV
  latents) round-trip instead of crashing core's SaveLatent. Each stream is
  written as its own sibling tensor (`latent_tensor`, `latent_tensor_1`, ...)
  with the stream count recorded in metadata; a non-nested latent produces a
  file byte-identical to what core itself would write.
"""

from __future__ import annotations

import hashlib
import json
import os

import safetensors.torch
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io

try:
    from comfy.nested_tensor import NestedTensor
except ImportError:  # older ComfyUI builds without nested-tensor support
    NestedTensor = None

_CAT = "latent/io"

_LEGACY_SD1_SCALE = 0.18215


def _is_nested(tensor) -> bool:
    if NestedTensor is not None and isinstance(tensor, NestedTensor):
        return True
    return bool(getattr(tensor, "is_nested", False)) and hasattr(tensor, "unbind")


def _nested_stream_tensors(prefix: str, streams: list) -> dict:
    out = {}
    for i, t in enumerate(streams):
        key = prefix if i == 0 else f"{prefix}_{i}"
        out[key] = t.detach().cpu().contiguous()
    return out


def _collect_nested_streams(tensors: dict, prefix: str, count: int) -> list:
    streams = []
    for i in range(count):
        key = prefix if i == 0 else f"{prefix}_{i}"
        if key not in tensors:
            raise ValueError(
                f"latent_io: expected {count} '{prefix}' streams but '{key}' is missing"
            )
        streams.append(tensors[key])
    return streams


def build_latent_tensors_and_metadata(samples: dict) -> tuple[dict, dict]:
    """Split a LATENT dict into a safetensors tensor dict + string metadata."""
    tensors = {}
    metadata = {"latent_format_version_0": "1"}

    latent_tensor = samples["samples"]
    if _is_nested(latent_tensor):
        streams = latent_tensor.unbind()
        tensors.update(_nested_stream_tensors("latent_tensor", streams))
        metadata["nested_streams"] = str(len(streams))
        metadata["verigen_latent_schema"] = "1"
    else:
        tensors["latent_tensor"] = latent_tensor.detach().cpu().contiguous()

    noise_mask = samples.get("noise_mask")
    if noise_mask is not None:
        if _is_nested(noise_mask):
            mask_streams = noise_mask.unbind()
            tensors.update(_nested_stream_tensors("noise_mask", mask_streams))
            metadata["nested_noise_mask_streams"] = str(len(mask_streams))
            metadata["verigen_latent_schema"] = "1"
        else:
            tensors["noise_mask"] = noise_mask.detach().cpu().contiguous()
        metadata["has_noise_mask"] = "1"

    batch_index = samples.get("batch_index")
    if batch_index is not None:
        metadata["batch_index"] = json.dumps(batch_index)

    latent_type = samples.get("type")
    if latent_type is not None:
        metadata["latent_type"] = latent_type

    return tensors, metadata


def latent_from_tensors_and_metadata(tensors: dict, metadata: dict | None) -> dict:
    """Inverse of build_latent_tensors_and_metadata."""
    metadata = metadata or {}
    is_legacy = "latent_format_version_0" not in metadata

    nested_streams = int(metadata.get("nested_streams") or "0")
    if nested_streams > 1:
        if NestedTensor is None:
            raise RuntimeError(
                "latent_io: file contains a nested latent (nested_streams="
                f"{nested_streams}) but this ComfyUI build has no comfy.nested_tensor"
            )
        streams = _collect_nested_streams(tensors, "latent_tensor", nested_streams)
        if is_legacy:
            streams = [t.float() * (1.0 / _LEGACY_SD1_SCALE) for t in streams]
        samples_tensor = NestedTensor(streams)
    else:
        samples_tensor = tensors["latent_tensor"]
        if is_legacy:
            samples_tensor = samples_tensor.float() * (1.0 / _LEGACY_SD1_SCALE)

    latent = {"samples": samples_tensor}

    mask_streams = int(metadata.get("nested_noise_mask_streams") or "0")
    if mask_streams > 1:
        if NestedTensor is None:
            raise RuntimeError(
                "latent_io: file contains a nested noise_mask (nested_noise_mask_streams="
                f"{mask_streams}) but this ComfyUI build has no comfy.nested_tensor"
            )
        latent["noise_mask"] = NestedTensor(_collect_nested_streams(tensors, "noise_mask", mask_streams))
    elif "noise_mask" in tensors:
        latent["noise_mask"] = tensors["noise_mask"]

    if metadata.get("batch_index"):
        latent["batch_index"] = json.loads(metadata["batch_index"])
    if metadata.get("latent_type"):
        latent["type"] = metadata["latent_type"]
    return latent


def _fingerprint_file(path: str):
    try:
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.hexdigest()
    except OSError:
        return float("nan")


class SaveLatentPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_SaveLatentPath",
            display_name="Save Latent (Path)",
            search_aliases=["export latent"],
            category=_CAT,
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input(
                    "path", default="",
                    tooltip="Absolute file path to write the latent to (safetensors). "
                    "\".latent\" is appended if the path has no extension.",
                ),
                io.Boolean.Input(
                    "create_dirs", default=True,
                    tooltip="Create parent directories if they don't exist.",
                ),
            ],
            outputs=[io.Latent.Output()],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples, path, create_dirs=True) -> io.NodeOutput:
        if not os.path.splitext(path)[1]:
            path = f"{path}.latent"
        tensors, metadata = build_latent_tensors_and_metadata(samples)
        parent = os.path.dirname(path)
        if parent and create_dirs:
            os.makedirs(parent, exist_ok=True)
        safetensors.torch.save_file(tensors, path, metadata=metadata)
        return io.NodeOutput(samples, ui={"text": [f"wrote {path}"]})


class LoadLatentPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_LoadLatentPath",
            display_name="Load Latent (Path)",
            search_aliases=["import latent"],
            category=_CAT,
            inputs=[
                io.String.Input("path", default="", tooltip="Absolute path of a .latent/.safetensors file to read."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoadLatentPath: file not found: {path!r}")
        tensors = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        samples = latent_from_tensors_and_metadata(tensors, metadata)
        return io.NodeOutput(samples)

    @classmethod
    def fingerprint_inputs(cls, path):
        return _fingerprint_file(path)


class SaveLatent(io.ComfyNode):
    """Faithful mirror of core SaveLatent (nodes.py), routed through the nested-aware codec."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_SaveLatent",
            display_name="Save Latent",
            search_aliases=["export latent", "save latent"],
            category=_CAT,
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input("filename_prefix", default="latents/ComfyUI"),
            ],
            outputs=[io.Latent.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, samples, filename_prefix="latents/ComfyUI") -> io.NodeOutput:
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )

        metadata = None
        if not args.disable_metadata:
            prompt_info = json.dumps(cls.hidden.prompt) if cls.hidden.prompt is not None else ""
            metadata = {"prompt": prompt_info}
            if cls.hidden.extra_pnginfo is not None:
                for x in cls.hidden.extra_pnginfo:
                    metadata[x] = json.dumps(cls.hidden.extra_pnginfo[x])

        tensors, latent_metadata = build_latent_tensors_and_metadata(samples)
        if metadata is not None:
            metadata.update(latent_metadata)
        else:
            metadata = latent_metadata

        file = f"{filename}_{counter:05}_.latent"
        results = [{"filename": file, "subfolder": subfolder, "type": "output"}]

        safetensors.torch.save_file(tensors, os.path.join(full_output_folder, file), metadata=metadata)
        return io.NodeOutput(samples, ui={"latents": results})


class LoadLatent(io.ComfyNode):
    """Faithful mirror of core LoadLatent (nodes.py), routed through the nested-aware codec."""

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".latent")
        ]
        return io.Schema(
            node_id="verinuddle_LoadLatent",
            display_name="Load Latent",
            search_aliases=["import latent", "open latent"],
            category=_CAT,
            inputs=[
                io.Combo.Input("latent", options=sorted(files)),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent) -> io.NodeOutput:
        latent_path = folder_paths.get_annotated_filepath(latent)
        tensors = safetensors.torch.load_file(latent_path, device="cpu")
        with safetensors.safe_open(latent_path, framework="pt") as f:
            metadata = f.metadata()
        samples = latent_from_tensors_and_metadata(tensors, metadata)
        return io.NodeOutput(samples)

    @classmethod
    def fingerprint_inputs(cls, latent):
        return _fingerprint_file(folder_paths.get_annotated_filepath(latent))

    @classmethod
    def validate_inputs(cls, latent):
        if not folder_paths.exists_annotated_filepath(latent):
            return f"Invalid latent file: {latent}"
        return True
