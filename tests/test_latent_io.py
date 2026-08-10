"""Tests for the latent_io codec (build_latent_tensors_and_metadata /
latent_from_tensors_and_metadata).

These exercise the pure codec helpers directly, without booting ComfyUI. They
import latent_io.py as a standalone module (not via the verinuddle package)
so that importing it doesn't pull in verinuddle/__init__.py, which
transitively imports comfy_api/folder_paths and probes for a ComfyUI install.

Run with:
    pytest tests/test_latent_io.py
"""

import importlib.util
import json
import os
import tempfile

import pytest
import safetensors.torch
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LATENT_IO_PATH = os.path.join(_HERE, "..", "src", "verinuddle", "latent_io.py")


def _load_latent_io():
    spec = importlib.util.spec_from_file_location("latent_io_under_test", _LATENT_IO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


latent_io = _load_latent_io()
NestedTensor = latent_io.NestedTensor


def _save_and_load(tensors, metadata):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.latent")
        safetensors.torch.save_file(tensors, path, metadata=metadata)
        loaded_tensors = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt") as f:
            loaded_metadata = f.metadata()
        return loaded_tensors, loaded_metadata


def test_h3_roundtrip():
    video = torch.randn(1, 24, 12, 48, 84)
    audio = torch.randn(1, 32, 2, 207)
    samples = {"samples": NestedTensor((video, audio))}

    tensors, metadata = latent_io.build_latent_tensors_and_metadata(samples)
    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)
    latent = latent_io.latent_from_tensors_and_metadata(loaded_tensors, loaded_metadata)

    result = latent["samples"]
    assert result.is_nested
    streams = result.unbind()
    assert len(streams) == 2
    assert torch.equal(streams[0], video)
    assert torch.equal(streams[1], audio)
    assert streams[0].dtype == video.dtype
    assert streams[1].dtype == audio.dtype


def test_nested_noise_mask_independent_stream_count():
    samples_tensor = torch.randn(1, 4, 8, 8)
    mask_a = torch.rand(1, 1, 8, 8)
    mask_b = torch.rand(1, 1, 8, 8)
    mask_c = torch.rand(1, 1, 8, 8)
    samples = {
        "samples": samples_tensor,
        "noise_mask": NestedTensor((mask_a, mask_b, mask_c)),
    }

    tensors, metadata = latent_io.build_latent_tensors_and_metadata(samples)
    assert metadata["nested_noise_mask_streams"] == "3"
    assert "nested_streams" not in metadata

    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)
    latent = latent_io.latent_from_tensors_and_metadata(loaded_tensors, loaded_metadata)

    assert torch.equal(latent["samples"], samples_tensor)
    mask = latent["noise_mask"]
    assert mask.is_nested
    streams = mask.unbind()
    assert torch.equal(streams[0], mask_a)
    assert torch.equal(streams[1], mask_b)
    assert torch.equal(streams[2], mask_c)


def test_plain_tensor_matches_core_schema():
    samples_tensor = torch.randn(1, 4, 8, 8)
    samples = {"samples": samples_tensor}

    tensors, metadata = latent_io.build_latent_tensors_and_metadata(samples)

    assert set(tensors.keys()) == {"latent_tensor"}
    assert metadata == {"latent_format_version_0": "1"}
    assert torch.equal(tensors["latent_tensor"], samples_tensor)


def test_forward_interop_core_load_latent_reads_stream_zero():
    video = torch.randn(1, 24, 4, 4, 4)
    audio = torch.randn(1, 32, 2, 10)
    samples = {"samples": NestedTensor((video, audio))}
    tensors, metadata = latent_io.build_latent_tensors_and_metadata(samples)
    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)

    # Mirrors core LoadLatent.load()'s logic (nodes.py) without importing nodes.py.
    multiplier = 1.0
    if "latent_format_version_0" not in loaded_metadata:
        multiplier = 1.0 / 0.18215
    core_samples = loaded_tensors["latent_tensor"].float() * multiplier
    assert torch.equal(core_samples, video.float())


def test_backward_interop_core_written_file():
    samples_tensor = torch.randn(1, 4, 8, 8)
    tensors = {"latent_tensor": samples_tensor.contiguous()}
    metadata = {"latent_format_version_0": "1"}
    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)

    latent = latent_io.latent_from_tensors_and_metadata(loaded_tensors, loaded_metadata)
    assert torch.equal(latent["samples"], samples_tensor)


def test_legacy_file_without_version_marker_is_rescaled():
    samples_tensor = torch.randn(1, 4, 8, 8)
    tensors = {"latent_tensor": samples_tensor.contiguous()}
    loaded_tensors, loaded_metadata = _save_and_load(tensors, {})

    latent = latent_io.latent_from_tensors_and_metadata(loaded_tensors, loaded_metadata)
    expected = samples_tensor.float() * (1.0 / 0.18215)
    assert torch.allclose(latent["samples"], expected)


def test_corrupt_nested_streams_count_raises_clear_error():
    video = torch.randn(1, 4, 4, 4)
    tensors = {"latent_tensor": video}
    metadata = {"latent_format_version_0": "1", "nested_streams": "3"}

    with pytest.raises(ValueError, match="latent_tensor_1"):
        latent_io.latent_from_tensors_and_metadata(tensors, metadata)


def test_batch_index_and_type_roundtrip():
    samples_tensor = torch.randn(1, 4, 8, 8)
    samples = {"samples": samples_tensor, "batch_index": [0], "type": "video"}

    tensors, metadata = latent_io.build_latent_tensors_and_metadata(samples)
    assert json.loads(metadata["batch_index"]) == [0]
    assert metadata["latent_type"] == "video"

    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)
    latent = latent_io.latent_from_tensors_and_metadata(loaded_tensors, loaded_metadata)
    assert latent["batch_index"] == [0]
    assert latent["type"] == "video"
