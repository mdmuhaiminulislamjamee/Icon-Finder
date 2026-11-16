from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from detector.preprocessing import load_rgb, preprocess, load_template


@pytest.mark.parametrize("format", ["PNG", "JPEG", "WEBP", "BMP"])
def test_supported_formats_preserve_size(format):
    buffer = BytesIO()
    Image.new("RGB", (137, 81), "white").save(buffer, format=format)
    assert load_rgb(buffer.getvalue()).shape == (81, 137, 3)


@pytest.mark.parametrize("data", [b"", b"not an image", b"\x89PNG\r\n\x1a\ntruncated"])
def test_corrupted_images_rejected(data):
    with pytest.raises(ValueError):
        load_rgb(data)


def test_transparency_and_array_validation():
    buffer = BytesIO()
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(buffer, format="PNG")
    assert tuple(load_rgb(buffer.getvalue())[0, 0]) == (238, 238, 238)
    with pytest.raises(ValueError):
        load_rgb(np.zeros((20, 20), dtype=np.uint8))
    with pytest.raises(ValueError):
        load_rgb(np.zeros((2, 2, 3), dtype=float))


def test_preprocessing_maps_and_template_cache(manager):
    path = manager.load("generic").directory / "icons/play_light.png"
    template = load_template(path)
    assert template is load_template(path)
    prepared = preprocess(load_rgb(path))
    assert prepared.rgb.shape == (24, 24, 3)
    for array in (prepared.gray, prepared.edges, prepared.binary, prepared.normalized):
        assert array.shape == (24, 24) and array.dtype == np.uint8
    assert np.count_nonzero(prepared.edges) > 0
    assert set(np.unique(prepared.binary)) == {0, 255}


def test_unsupported_and_oversize_images(monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="GIF")
    with pytest.raises(ValueError, match="PNG"):
        load_rgb(buffer.getvalue())
    monkeypatch.setattr("detector.preprocessing.MAX_PIXELS", 100)
    buffer = BytesIO()
    Image.new("RGB", (20, 20), "white").save(buffer, format="PNG")
    with pytest.raises(ValueError, match="40 megapixels"):
        load_rgb(buffer.getvalue())
