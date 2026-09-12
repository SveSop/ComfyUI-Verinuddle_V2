"""Conditioning import/export nodes and the shared safetensors schema they use.

Unlike LATENT (one fixed `samples` tensor plus a handful of known optional
keys, see latent_io.py), CONDITIONING is an open-ended structure:
`list[[tensor, dict]]`, where the dict's extra values can themselves be
tensors, ints, strings, or lists of dicts (e.g. MiniMax H3's
`minimax_keyframes`/`minimax_refs`, each a list of dicts mixing
`resolved_frame_index`/`kind` (int/str) with `latent`/`audio_latent`
(tensor)). There's no fixed key list to special-case, so this codec walks the
structure generically: every tensor leaf is pulled into a flat safetensors
payload and replaced in place by a placeholder, and the resulting JSON-safe
skeleton is stored as safetensors metadata.

Anything that isn't a tensor, a JSON-safe scalar, or a list/dict/tuple
container (e.g. a ControlNet object) raises a clear TypeError rather than
being silently pickled -- conditioning as produced by core and MiniMax H3
never contains such values, so this is a deliberate scope limit, not a gap.
A comfy.nested_tensor.NestedTensor leaf raises for the same reason: nothing
puts one inside CONDITIONING today (AV latents carry NestedTensor samples,
not conditioning), so round-tripping it is out of scope rather than silently
wrong.
"""

from __future__ import annotations

import hashlib
import json
import os

import safetensors.torch
import torch
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

try:
    from comfy.nested_tensor import NestedTensor
except ImportError:  # older ComfyUI builds without nested-tensor support
    NestedTensor = None

_CAT = "conditioning/io"

_TENSOR_KEY = "__tensor__"
_TUPLE_KEY = "__tuple__"


def _is_nested(tensor) -> bool:
    if NestedTensor is not None and isinstance(tensor, NestedTensor):
        return True
    return bool(getattr(tensor, "is_nested", False)) and hasattr(tensor, "unbind")


def _extract_tensors(value, tensors: list, path: str):
    if _is_nested(value):
        raise RuntimeError(
            f"conditioning_io: nested tensor at {path!r} is not supported in CONDITIONING"
        )
    if isinstance(value, torch.Tensor):
        idx = len(tensors)
        tensors.append(value.detach().cpu().contiguous())
        return {_TENSOR_KEY: idx}
    if isinstance(value, dict):
        return {k: _extract_tensors(v, tensors, f"{path}.{k}") for k, v in value.items()}
    if isinstance(value, list):
        return [_extract_tensors(v, tensors, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, tuple):
        return {_TUPLE_KEY: [_extract_tensors(v, tensors, f"{path}[{i}]") for i, v in enumerate(value)]}
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise TypeError(f"conditioning_io: unsupported value type {type(value)!r} at {path!r}")


def _restore_tensors(value, tensors: list):
    if isinstance(value, dict):
        if set(value.keys()) == {_TENSOR_KEY}:
            return tensors[value[_TENSOR_KEY]]
        if set(value.keys()) == {_TUPLE_KEY}:
            return tuple(_restore_tensors(v, tensors) for v in value[_TUPLE_KEY])
        return {k: _restore_tensors(v, tensors) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_tensors(v, tensors) for v in value]
    return value


def build_conditioning_tensors_and_metadata(conditioning) -> tuple[dict, dict]:
    """Split a CONDITIONING value into a safetensors tensor dict + string metadata."""
    tensors: list = []
    skeleton = _extract_tensors(conditioning, tensors, "conditioning")
    tensor_dict = {f"tensor_{i}": t for i, t in enumerate(tensors)}
    metadata = {
        "conditioning_format_version_0": "1",
        "conditioning_schema": json.dumps(skeleton),
    }
    return tensor_dict, metadata


def conditioning_from_tensors_and_metadata(tensors: dict, metadata: dict | None):
    """Inverse of build_conditioning_tensors_and_metadata."""
    metadata = metadata or {}
    if "conditioning_schema" not in metadata:
        raise ValueError("conditioning_io: file has no 'conditioning_schema' metadata")
    skeleton = json.loads(metadata["conditioning_schema"])
    tensor_list = []
    i = 0
    while f"tensor_{i}" in tensors:
        tensor_list.append(tensors[f"tensor_{i}"])
        i += 1
    return _restore_tensors(skeleton, tensor_list)


def _fingerprint_file(path: str):
    try:
        m = hashlib.sha256()
        with open(path, "rb") as f:
            m.update(f.read())
        return m.hexdigest()
    except OSError:
        return float("nan")


class SaveConditioningPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_SaveConditioningPath",
            display_name="Save Conditioning (Path)",
            search_aliases=["export conditioning"],
            category=_CAT,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input(
                    "path", default="",
                    tooltip="Absolute file path to write the conditioning to (safetensors). "
                    "\".conditioning\" is appended if the path has no extension.",
                ),
                io.Boolean.Input(
                    "create_dirs", default=True,
                    tooltip="Create parent directories if they don't exist.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, conditioning, path, create_dirs=True) -> io.NodeOutput:
        if not os.path.splitext(path)[1]:
            path = f"{path}.conditioning"
        tensors, metadata = build_conditioning_tensors_and_metadata(conditioning)
        parent = os.path.dirname(path)
        if parent and create_dirs:
            os.makedirs(parent, exist_ok=True)
        safetensors.torch.save_file(tensors, path, metadata=metadata)
        return io.NodeOutput(conditioning, ui={"text": [f"wrote {path}"]})


class LoadConditioningPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_LoadConditioningPath",
            display_name="Load Conditioning (Path)",
            search_aliases=["import conditioning"],
            category=_CAT,
            inputs=[
                io.String.Input("path", default="", tooltip="Absolute path of a .conditioning/.safetensors file to read."),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"LoadConditioningPath: file not found: {path!r}")
        tensors = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        conditioning = conditioning_from_tensors_and_metadata(tensors, metadata)
        return io.NodeOutput(conditioning)

    @classmethod
    def fingerprint_inputs(cls, path):
        return _fingerprint_file(path)


class BackupConditioningPath(io.ComfyNode):
    """Save+Load (Path) merged behind one switch, with graph execution optimized
    in both directions:

    - ``conditioning`` is declared lazy so that in Load mode (save=False),
      ComfyUI's execution engine never schedules whatever upstream graph
      would have produced it (e.g. an expensive MiniMax H3 encode) -- see
      check_lazy_status.
    - ``stop_here`` (Save mode only) returns an ExecutionBlocker in place of
      the real output, so nothing wired downstream of this node executes
      either -- see execute. This lets a workflow be split into
      independently-queueable phases (e.g. MiniMax H3 encode, then a
      separate run for sampling) around a saved checkpoint.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_BackupConditioningPath",
            display_name="Backup Conditioning (Path)",
            search_aliases=["save or load conditioning", "conditioning backup"],
            category=_CAT,
            inputs=[
                io.Boolean.Input(
                    "save", default=True,
                    label_on="Save", label_off="Load",
                    tooltip="Save: write conditioning to path and pass it through unchanged. "
                    "Load: ignore conditioning (its upstream graph is not executed) and "
                    "output what's stored at path instead.",
                ),
                io.Conditioning.Input("conditioning", lazy=True),
                io.String.Input(
                    "path", default="",
                    tooltip="Absolute file path to read/write the conditioning (safetensors). "
                    "\".conditioning\" is appended if the path has no extension.",
                ),
                io.Boolean.Input(
                    "create_dirs", default=True,
                    tooltip="Create parent directories if they don't exist (Save mode only).",
                ),
                io.Boolean.Input(
                    "stop_here", default=False,
                    tooltip="Save mode only: after saving, block every node downstream of "
                    "this one's output from executing (silently -- no error). Lets you "
                    "split an expensive workflow into independently-queueable phases "
                    "(e.g. MiniMax H3 encode, then a separate run for sampling). "
                    "Ignored in Load mode, where the loaded conditioning always passes through.",
                ),
            ],
            outputs=[io.Conditioning.Output()],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(cls, save, conditioning=None, path=None, create_dirs=None, stop_here=None):
        if save and conditioning is None:
            return ["conditioning"]
        return []

    @staticmethod
    def _normalize_path(path):
        if not os.path.splitext(path)[1]:
            return f"{path}.conditioning"
        return path

    @classmethod
    def execute(cls, save, conditioning, path, create_dirs=True, stop_here=False) -> io.NodeOutput:
        path = cls._normalize_path(path)
        if save:
            tensors, metadata = build_conditioning_tensors_and_metadata(conditioning)
            parent = os.path.dirname(path)
            if parent and create_dirs:
                os.makedirs(parent, exist_ok=True)
            safetensors.torch.save_file(tensors, path, metadata=metadata)
            if stop_here:
                return io.NodeOutput(ExecutionBlocker(None), ui={"text": [f"wrote {path}"]})
            return io.NodeOutput(conditioning, ui={"text": [f"wrote {path}"]})
        if not os.path.isfile(path):
            raise FileNotFoundError(f"BackupConditioningPath: file not found: {path!r}")
        tensors = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata()
        return io.NodeOutput(conditioning_from_tensors_and_metadata(tensors, metadata))

    @classmethod
    def fingerprint_inputs(cls, save, conditioning=None, path=None, create_dirs=None, stop_here=None):
        if not save:
            return _fingerprint_file(cls._normalize_path(path))
        return float("nan")  # Save mode: always re-run so the file gets written.


class SaveConditioning(io.ComfyNode):
    """Faithful mirror of verinuddle SaveLatent, routed through the conditioning codec."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_SaveConditioning",
            display_name="Save Conditioning",
            search_aliases=["export conditioning", "save conditioning"],
            category=_CAT,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.String.Input("filename_prefix", default="conditionings/ComfyUI"),
            ],
            outputs=[io.Conditioning.Output()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, conditioning, filename_prefix="conditionings/ComfyUI") -> io.NodeOutput:
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

        tensors, conditioning_metadata = build_conditioning_tensors_and_metadata(conditioning)
        if metadata is not None:
            metadata.update(conditioning_metadata)
        else:
            metadata = conditioning_metadata

        file = f"{filename}_{counter:05}_.conditioning"
        results = [{"filename": file, "subfolder": subfolder, "type": "output"}]

        safetensors.torch.save_file(tensors, os.path.join(full_output_folder, file), metadata=metadata)
        return io.NodeOutput(conditioning, ui={"conditionings": results})


class LoadConditioning(io.ComfyNode):
    """Faithful mirror of verinuddle LoadLatent, routed through the conditioning codec."""

    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, f)) and f.endswith(".conditioning")
        ]
        return io.Schema(
            node_id="verinuddle_LoadConditioning",
            display_name="Load Conditioning",
            search_aliases=["import conditioning", "open conditioning"],
            category=_CAT,
            inputs=[
                io.Combo.Input("conditioning", options=sorted(files)),
            ],
            outputs=[io.Conditioning.Output()],
        )

    @classmethod
    def execute(cls, conditioning) -> io.NodeOutput:
        conditioning_path = folder_paths.get_annotated_filepath(conditioning)
        tensors = safetensors.torch.load_file(conditioning_path, device="cpu")
        with safetensors.safe_open(conditioning_path, framework="pt") as f:
            metadata = f.metadata()
        value = conditioning_from_tensors_and_metadata(tensors, metadata)
        return io.NodeOutput(value)

    @classmethod
    def fingerprint_inputs(cls, conditioning):
        return _fingerprint_file(folder_paths.get_annotated_filepath(conditioning))

    @classmethod
    def validate_inputs(cls, conditioning):
        if not folder_paths.exists_annotated_filepath(conditioning):
            return f"Invalid conditioning file: {conditioning}"
        return True
