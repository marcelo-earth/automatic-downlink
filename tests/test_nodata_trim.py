"""Tests for the Sentinel no-data border trim in the triage loop."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.triage.loop import _trim_nodata_pair
from tests.conftest import noisy_image


def with_black_border(image: Image.Image, left: int = 0, right: int = 0) -> Image.Image:
    arr = np.asarray(image).copy()
    if left:
        arr[:, :left] = 0
    if right:
        arr[:, -right:] = 0
    return Image.fromarray(arr, "RGB")


def test_clean_image_is_untouched() -> None:
    image = noisy_image()
    trimmed, swir = _trim_nodata_pair(image)
    assert trimmed is image
    assert swir is None


def test_black_side_strip_is_removed() -> None:
    image = with_black_border(noisy_image(), left=40)
    trimmed, _ = _trim_nodata_pair(image)
    assert trimmed.size == (128 - 40, 128)


def test_swir_is_cropped_with_same_bbox() -> None:
    rgb = with_black_border(noisy_image(seed=1), right=30)
    swir = with_black_border(noisy_image(seed=2), right=30)
    trimmed, trimmed_swir = _trim_nodata_pair(rgb, swir)
    assert trimmed.size == trimmed_swir.size == (128 - 30, 128)


def test_dark_interior_is_not_cropped() -> None:
    # A dark burn scar in the middle must survive the trim.
    arr = np.asarray(noisy_image()).copy()
    arr[40:90, 40:90] = 0
    image = Image.fromarray(arr, "RGB")
    trimmed, _ = _trim_nodata_pair(image)
    assert trimmed.size == image.size


def test_mostly_nodata_frame_is_kept_whole() -> None:
    # Cropping away more than 65% of the frame is riskier than keeping it.
    image = with_black_border(noisy_image(), left=100)
    trimmed, _ = _trim_nodata_pair(image)
    assert trimmed.size == image.size


def test_tiny_border_noise_is_ignored() -> None:
    image = with_black_border(noisy_image(), left=2)
    trimmed, _ = _trim_nodata_pair(image)
    assert trimmed.size == image.size
