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

import gc
import hashlib
import json
import os

import safetensors.torch
import folder_paths
from comfy.cli_args import args
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

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
        latent["noise_mask"] = NestedTensor(
            _collect_nested_streams(tensors, "noise_mask", mask_streams)
        )
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


def _under_output(path):
    """Resolve a path inside ComfyUI's output folder.

    Relative paths are resolved against the ComfyUI output directory.
    Absolute paths are allowed only when they already live inside output/.
    """
    root = os.path.realpath(folder_paths.get_output_directory())
    p = (path or "").strip().strip('"').strip("'") or "h3_latent/Clip"

    resolved = os.path.realpath(
        p if os.path.isabs(p) else os.path.join(root, p)
    )

    if resolved != root and not resolved.startswith(root + os.sep):
        return None

    return resolved


def _resolve_latent_path(path, index=0):
    """Resolve an indexed latent slot from a ComfyUI output filename prefix.

    Example:

        path  = h3_latent/Clip
        index = 1

    resolves to:

        output/h3_latent/Clip_00001.safetensors

    Index 0 is handled by LoadLatent itself and means no latent.
    """
    root = os.path.realpath(folder_paths.get_output_directory())
    p = (path or "").strip().strip('"').strip("'") or "h3_latent/Clip"

    # Resolve the user path relative to ComfyUI's output directory.
    resolved = os.path.realpath(
        p if os.path.isabs(p) else os.path.join(root, p)
    )

    if resolved != root and not resolved.startswith(root + os.sep):
        raise FileNotFoundError(
            "Verinuddle: path must stay inside the ComfyUI output folder."
        )

    idx = int(index)

    if idx <= 0:
        raise FileNotFoundError(
            "Verinuddle: index 0 does not load a file."
        )

    # If the supplied path is already an existing file, load that file.
    if os.path.isfile(resolved):
        return resolved

    # Treat the supplied path as a filename prefix.
    folder = os.path.dirname(resolved)
    prefix = os.path.basename(resolved)

    if not os.path.isdir(folder):
        raise FileNotFoundError(
            "Verinuddle: folder does not exist: %s" % folder
        )

    endings = (
        f"_{idx:05d}.safetensors",
        f"_clip{idx:03d}.safetensors",
    )

    files = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.startswith(prefix + "_")
        and name.endswith(endings)
    ]

    if not files:
        near = [
            name
            for name in os.listdir(folder)
            if name.startswith(prefix + "_")
            and name.endswith(f"_{idx:05d}_.safetensors")
        ]

        hint = ""
        if near:
            hint = (
                f" Found {near[0]}, which is an auto-numbered save "
                f"(trailing underscore = numbered by RUN)."
            )

        raise FileNotFoundError(
            "Verinuddle: no saved latent for index %d "
            "(no %s_%05d.safetensors in %s).%s"
            % (idx, prefix, idx, folder, hint)
        )

    return max(files, key=os.path.getmtime)


def _write_safetensors(path, tensors, metadata):
    """Safely overwrite a safetensors file on Windows.

    safetensors load_file can leave a file memory-mapped. Writing a temporary
    sibling and replacing the target avoids Windows sharing violations when
    an indexed slot is overwritten.
    """
    tmp = path + ".tmp"

    try:
        safetensors.torch.save_file(
            tensors,
            tmp,
            metadata=metadata,
        )

        try:
            os.replace(tmp, path)
        except OSError:
            gc.collect()
            os.replace(tmp, path)

    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


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
                    "path",
                    default="",
                    tooltip=(
                        "Absolute file path to write the latent to (safetensors). "
                        "\".latent\" is appended if the path has no extension."
                    ),
                ),
                io.Boolean.Input(
                    "create_dirs",
                    default=True,
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

        safetensors.torch.save_file(
            tensors,
            path,
            metadata=metadata,
        )

        return io.NodeOutput(
            samples,
            ui={"text": [f"wrote {path}"]},
        )


class LoadLatentPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_LoadLatentPath",
            display_name="Load Latent (Path)",
            search_aliases=["import latent"],
            category=_CAT,
            inputs=[
                io.String.Input(
                    "path",
                    default="",
                    tooltip=(
                        "Absolute path of a .latent/.safetensors file to read."
                    ),
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, path) -> io.NodeOutput:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"LoadLatentPath: file not found: {path!r}"
            )

        tensors = safetensors.torch.load_file(
            path,
            device="cpu",
        )

        with safetensors.safe_open(
            path,
            framework="pt",
        ) as f:
            metadata = f.metadata()

        samples = latent_from_tensors_and_metadata(
            tensors,
            metadata,
        )

        return io.NodeOutput(samples)

    @classmethod
    def fingerprint_inputs(cls, path):
        return _fingerprint_file(path)


class BackupLatentPath(io.ComfyNode):
    """Save+Load (Path) merged behind one switch, with graph execution optimized
    in both directions:

    - ``samples`` is declared lazy so that in Load mode (save=False), ComfyUI's
      execution engine never schedules whatever upstream graph would have
      produced it -- see check_lazy_status.
    - ``stop_here`` (Save mode only) returns an ExecutionBlocker in place of the
      real output, so nothing wired downstream of this node executes either --
      see execute. This lets a workflow be split into independently-queueable
      phases (e.g. generation, then a separate run for frame interpolation)
      around a saved checkpoint.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_BackupLatentPath",
            display_name="Backup Latent (Path)",
            search_aliases=["save or load latent", "latent backup"],
            category=_CAT,
            inputs=[
                io.Boolean.Input(
                    "save",
                    default=True,
                    label_on="Save",
                    label_off="Load",
                    tooltip=(
                        "Save: write samples to path and pass them through unchanged. "
                        "Load: ignore samples (its upstream graph is not executed) "
                        "and output what's stored at path instead."
                    ),
                ),
                io.Latent.Input("samples", lazy=True),
                io.String.Input(
                    "path",
                    default="",
                    tooltip=(
                        "Absolute file path to read/write the latent (safetensors). "
                        "\".latent\" is appended if the path has no extension."
                    ),
                ),
                io.Boolean.Input(
                    "create_dirs",
                    default=True,
                    tooltip=(
                        "Create parent directories if they don't exist "
                        "(Save mode only)."
                    ),
                ),
                io.Boolean.Input(
                    "stop_here",
                    default=False,
                    tooltip=(
                        "Save mode only: after saving, block every node downstream "
                        "of this one's output from executing (silently -- no error). "
                        "Lets you split an expensive workflow into independently-"
                        "queueable phases (e.g. generation, then a separate run "
                        "for frame interpolation). Ignored in Load mode, where "
                        "the loaded latent always passes through."
                    ),
                ),
            ],
            outputs=[io.Latent.Output()],
            is_output_node=True,
        )

    @classmethod
    def check_lazy_status(
        cls,
        save,
        samples=None,
        path=None,
        create_dirs=None,
        stop_here=None,
    ):
        if save and samples is None:
            return ["samples"]
        return []

    @staticmethod
    def _normalize_path(path):
        if not os.path.splitext(path)[1]:
            return f"{path}.latent"
        return path

    @classmethod
    def execute(
        cls,
        save,
        samples,
        path,
        create_dirs=True,
        stop_here=False,
    ) -> io.NodeOutput:
        path = cls._normalize_path(path)

        if save:
            tensors, metadata = build_latent_tensors_and_metadata(samples)

            parent = os.path.dirname(path)
            if parent and create_dirs:
                os.makedirs(parent, exist_ok=True)

            safetensors.torch.save_file(
                tensors,
                path,
                metadata=metadata,
            )

            if stop_here:
                return io.NodeOutput(
                    ExecutionBlocker(None),
                    ui={"text": [f"wrote {path}"]},
                )

            return io.NodeOutput(
                samples,
                ui={"text": [f"wrote {path}"]},
            )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"BackupLatentPath: file not found: {path!r}"
            )

        tensors = safetensors.torch.load_file(
            path,
            device="cpu",
        )

        with safetensors.safe_open(
            path,
            framework="pt",
        ) as f:
            metadata = f.metadata()

        return io.NodeOutput(
            latent_from_tensors_and_metadata(
                tensors,
                metadata,
            )
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        save,
        samples=None,
        path=None,
        create_dirs=None,
        stop_here=None,
    ):
        if not save:
            return _fingerprint_file(
                cls._normalize_path(path)
            )

        return float("nan")


class SaveLatent(io.ComfyNode):
    """Save a LATENT using indexed H3 clip slots.

    index > 0:
        Save to an explicit slot such as Clip_00001.safetensors.
        Existing contents of that slot are overwritten.

    index = 0:
        Use ComfyUI's normal automatic counter, producing names such as
        Clip_00001_.safetensors.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_SaveLatent",
            display_name="Save Latent",
            search_aliases=["export latent", "save latent"],
            category=_CAT,
            inputs=[
                io.Latent.Input("samples"),
                io.String.Input(
                    "path",
                    default="h3_latent/Clip",
                    tooltip=(
                        "Path relative to the ComfyUI output folder. "
                        "For example, h3_latent/Clip saves into "
                        "output/h3_latent."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=1,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip=(
                        "Explicit clip slot. 1 writes Clip_00001.safetensors "
                        "and overwrites that slot. 0 uses ComfyUI's automatic "
                        "run counter and writes Clip_00001_.safetensors, etc."
                    ),
                ),
            ],
            outputs=[io.Latent.Output()],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        samples,
        path="h3_latent/Clip",
        index=1,
    ) -> io.NodeOutput:
        # Use ComfyUI's normal filename-prefix handling to determine the
        # output directory, filename prefix, and automatic counter.
        folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            path,
            folder_paths.get_output_directory(),
        )

        os.makedirs(folder, exist_ok=True)

        if int(index) > 0:
            # Explicit clip slot.
            # Example:
            #   path  = h3_latent/Clip
            #   index = 1
            #   -> output/h3_latent/Clip_00001.safetensors
            full_path = os.path.join(
                folder,
                f"{filename}_{int(index):05d}.safetensors",
            )
        else:
            # ComfyUI automatic run counter.
            # The trailing underscore deliberately distinguishes these files
            # from explicit clip slots.
            full_path = os.path.join(
                folder,
                f"{filename}_{counter:05d}_.safetensors",
            )

        # Preserve Verinuddle's existing nested-H3 serialization.
        tensors, metadata = build_latent_tensors_and_metadata(samples)

        # Use a temporary file + os.replace() so an indexed slot can safely
        # overwrite a file that may still be memory-mapped on Windows.
        _write_safetensors(
            full_path,
            tensors,
            metadata,
        )

        return io.NodeOutput(
            samples,
            ui={"text": [f"wrote {full_path}"]},
        )


class LoadLatent(io.ComfyNode):
    """Load a LATENT from an indexed H3 clip slot.

    index = 0:
        No previous clip. Returns None.

    index > 0:
        Loads the corresponding explicit slot, e.g.
        index 1 -> Clip_00001.safetensors
        index 2 -> Clip_00002.safetensors
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="verinuddle_LoadLatent",
            display_name="Load Latent",
            search_aliases=["import latent", "open latent"],
            category=_CAT,
            inputs=[
                io.String.Input(
                    "path",
                    default="h3_latent/Clip",
                    tooltip=(
                        "Path relative to the ComfyUI output folder. "
                        "For example, h3_latent/Clip searches "
                        "output/h3_latent."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=1,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip=(
                        "Clip slot to load. 1 loads Clip_00001.safetensors. "
                        "0 means no previous clip and loads nothing."
                    ),
                ),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(
        cls,
        path="h3_latent/Clip",
        index=1,
    ) -> io.NodeOutput:
        if int(index) <= 0:
            return io.NodeOutput(None)

        latent_path = _resolve_latent_path(
            path,
            index,
        )

        tensors = safetensors.torch.load_file(
            latent_path,
            device="cpu",
        )

        with safetensors.safe_open(
            latent_path,
            framework="pt",
        ) as f:
            metadata = f.metadata()

        # This reconstructs the full Verinuddle LATENT, including a nested
        # H3 video+audio NestedTensor when the saved file contains one.
        samples = latent_from_tensors_and_metadata(
            tensors,
            metadata,
        )

        return io.NodeOutput(samples)

    @classmethod
    def fingerprint_inputs(
        cls,
        path,
        index=1,
    ):
        if int(index) <= 0:
            return "disabled"

        try:
            latent_path = _resolve_latent_path(
                path,
                index,
            )
            return "%s:%d" % (
                latent_path,
                os.stat(latent_path).st_mtime_ns,
            )
        except Exception:
            # If the file doesn't exist yet, don't let ComfyUI cache a
            # permanently invalid result.
            return float("nan")

    @classmethod
    def validate_inputs(
        cls,
        path,
        index=1,
    ):
        if int(index) <= 0:
            return True

        try:
            _resolve_latent_path(
                path,
                index,
            )
            return True
        except FileNotFoundError as e:
            return str(e)

