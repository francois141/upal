"""Focused public-API and end-to-end tests for the UPAL inference package."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

import upal
from upal import UPAL, load_model
from upal.demo_utils import load_image


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_OUTPUTS = {
    "keypoints",
    "keypoint_scores",
    "keypoint_dispersity",
    "descriptors",
    "keypoint_heatmap",
    "line_distance_field",
    "lines",
}


@pytest.fixture(scope="module")
def model() -> UPAL:
    return load_model(ROOT / "weights/upal.tar", max_num_keypoints=256)


@pytest.fixture(scope="module")
def image() -> torch.Tensor:
    return load_image(ROOT / "assets/boat1.png", max_size=320)[1]


def test_public_api_and_version() -> None:
    assert upal.__version__ == version("upal")
    assert {
        "UPAL",
        "load_model",
        "detect_lines",
        "mutual_nearest_neighbors",
        "match_lines_from_endpoints",
    }.issubset(upal.__all__)


def test_extract_without_optional_line_postprocessing(model: UPAL, image: torch.Tensor) -> None:
    prediction = model.extract(image, lines=False)

    assert set(prediction) == EXPECTED_OUTPUTS
    assert prediction["keypoints"].shape == (256, 2)
    assert prediction["keypoint_scores"].shape == (256,)
    assert prediction["keypoint_dispersity"].shape == (256,)
    assert prediction["descriptors"].shape == (256, 128)
    assert prediction["keypoint_heatmap"].shape == image.shape[-2:]
    assert prediction["line_distance_field"].shape == image.shape[-2:]
    assert prediction["lines"].shape == (0, 2, 2)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    torch.testing.assert_close(
        torch.linalg.vector_norm(prediction["descriptors"], dim=1),
        torch.ones(256),
        rtol=1e-5,
        atol=1e-6,
    )


def test_hub_round_trip_is_exact(model: UPAL, image: torch.Tensor, tmp_path: Path) -> None:
    output = tmp_path / "hub-model"
    model.save_pretrained(output)

    assert json.loads((output / "config.json").read_text()) == {
        "line_neighborhood": 5.0,
        "max_num_keypoints": 256,
        "nms_radius": 3,
    }
    restored = UPAL.from_pretrained(output)
    with torch.inference_mode():
        expected = model(image)
        actual = restored(image)
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    model_file = output / "model.safetensors"
    incomplete_state = load_file(model_file)
    missing_key = next(key for key in incomplete_state if not key.endswith("num_batches_tracked"))
    del incomplete_state[missing_key]
    save_file(incomplete_state, model_file)
    with pytest.raises(RuntimeError, match="Missing key"):
        UPAL.from_pretrained(output)


def test_end_to_end_line_detection_and_description(model: UPAL, image: torch.Tensor) -> None:
    prediction = model.extract(
        image,
        lines=True,
        max_lines=20,
        min_line_length=20.0,
        max_line_distance=2.0,
    )
    lines = prediction["lines"]

    assert lines.ndim == 3 and lines.shape[1:] == (2, 2)
    assert 0 < len(lines) <= 20
    assert torch.isfinite(lines).all()

    descriptors = model.describe_lines(image, lines)
    assert descriptors.shape == (len(lines), 2, 128)
    assert torch.isfinite(descriptors).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(descriptors, dim=2),
        torch.ones((len(lines), 2)),
        rtol=1e-5,
        atol=1e-6,
    )
