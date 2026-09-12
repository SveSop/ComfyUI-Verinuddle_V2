"""Tests for the conditioning_io codec (build_conditioning_tensors_and_metadata /
conditioning_from_tensors_and_metadata).

These exercise the pure codec helpers directly, without booting ComfyUI. They
import conditioning_io.py as a standalone module (not via the verinuddle
package) so that importing it doesn't pull in verinuddle/__init__.py, which
transitively imports comfy_api/folder_paths and probes for a ComfyUI install.

Run with:
    pytest tests/test_conditioning_io.py
"""

import importlib.util
import os
import tempfile

import pytest
import safetensors.torch
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONDITIONING_IO_PATH = os.path.join(_HERE, "..", "src", "verinuddle", "conditioning_io.py")


def _load_conditioning_io():
    spec = importlib.util.spec_from_file_location("conditioning_io_under_test", _CONDITIONING_IO_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


conditioning_io = _load_conditioning_io()
NestedTensor = conditioning_io.NestedTensor


def _save_and_load(tensors, metadata):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.conditioning")
        safetensors.torch.save_file(tensors, path, metadata=metadata)
        loaded_tensors = safetensors.torch.load_file(path, device="cpu")
        with safetensors.safe_open(path, framework="pt") as f:
            loaded_metadata = f.metadata()
        return loaded_tensors, loaded_metadata


def _roundtrip(conditioning):
    tensors, metadata = conditioning_io.build_conditioning_tensors_and_metadata(conditioning)
    loaded_tensors, loaded_metadata = _save_and_load(tensors, metadata)
    return conditioning_io.conditioning_from_tensors_and_metadata(loaded_tensors, loaded_metadata)


def _assert_values_equal(va, vb):
    if isinstance(va, torch.Tensor):
        assert torch.equal(va, vb)
    elif isinstance(va, dict):
        assert va.keys() == vb.keys()
        for key in va:
            _assert_values_equal(va[key], vb[key])
    elif isinstance(va, (list, tuple)):
        assert len(va) == len(vb)
        for item_a, item_b in zip(va, vb):
            _assert_values_equal(item_a, item_b)
    else:
        assert va == vb


def _assert_conditioning_equal(a, b):
    assert len(a) == len(b)
    for (cond_a, extra_a), (cond_b, extra_b) in zip(a, b):
        assert torch.equal(cond_a, cond_b)
        _assert_values_equal(extra_a, extra_b)


def test_plain_text_conditioning_roundtrip():
    cond_tensor = torch.randn(1, 77, 768)
    pooled = torch.randn(1, 768)
    conditioning = [[cond_tensor, {"pooled_output": pooled}]]

    result = _roundtrip(conditioning)
    _assert_conditioning_equal(result, conditioning)


def test_minimax_keyframes_roundtrip():
    cond_tensor = torch.randn(1, 32, 4096)
    keyframes = [
        {"resolved_frame_index": 0, "latent": torch.randn(1, 24, 2, 8, 8)},
        {"resolved_frame_index": 121, "latent": torch.randn(1, 24, 2, 8, 8)},
    ]
    conditioning = [[cond_tensor, {"minimax_keyframes": keyframes}]]

    result = _roundtrip(conditioning)
    _assert_conditioning_equal(result, conditioning)
    restored_keyframes = result[0][1]["minimax_keyframes"]
    assert restored_keyframes[0]["resolved_frame_index"] == 0
    assert restored_keyframes[1]["resolved_frame_index"] == 121
    assert torch.equal(restored_keyframes[0]["latent"], keyframes[0]["latent"])


def test_minimax_refs_mixed_entries_roundtrip():
    cond_tensor = torch.randn(1, 32, 4096)
    refs = [
        {"kind": "image", "latent_h": 48, "latent_w": 84, "latent": torch.randn(1, 24, 1, 48, 84)},
        {"kind": "audio", "ref_audio_t": 40, "audio_latent": torch.randn(1, 32, 2, 40)},
        {"kind": "video", "latent_t": 12, "latent_h": 48, "latent_w": 84, "ref_audio_t": 0,
         "latent": torch.randn(1, 24, 12, 48, 84), "audio_latent": None},
    ]
    conditioning = [[cond_tensor, {"minimax_refs": refs}]]

    result = _roundtrip(conditioning)
    _assert_conditioning_equal(result, conditioning)
    restored_refs = result[0][1]["minimax_refs"]
    assert restored_refs[1]["kind"] == "audio"
    assert restored_refs[2]["audio_latent"] is None


def test_tuple_valued_extra_key_roundtrip():
    cond_tensor = torch.randn(1, 8, 16)
    conditioning = [[cond_tensor, {"area": (8, 8, 0, 0)}]]

    result = _roundtrip(conditioning)
    assert result[0][1]["area"] == (8, 8, 0, 0)
    assert isinstance(result[0][1]["area"], tuple)


def test_unsupported_leaf_type_raises_type_error():
    cond_tensor = torch.randn(1, 8, 16)

    class NotSerializable:
        pass

    conditioning = [[cond_tensor, {"control": NotSerializable()}]]

    with pytest.raises(TypeError, match="control"):
        conditioning_io.build_conditioning_tensors_and_metadata(conditioning)


def test_nested_tensor_leaf_raises_runtime_error():
    cond_tensor = torch.randn(1, 8, 16)
    nested = NestedTensor((torch.randn(1, 4, 4), torch.randn(1, 2, 2)))
    conditioning = [[cond_tensor, {"minimax_keyframes": [{"latent": nested}]}]]

    with pytest.raises(RuntimeError, match="nested tensor"):
        conditioning_io.build_conditioning_tensors_and_metadata(conditioning)


def test_missing_schema_metadata_raises_clear_error():
    with pytest.raises(ValueError, match="conditioning_schema"):
        conditioning_io.conditioning_from_tensors_and_metadata({}, {})
