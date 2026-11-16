"""Image decoding and shared, deterministic classical preprocessing."""
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
MAX_PIXELS = 40_000_000
MAX_FILE_BYTES = 80_000_000


def load_rgb(source, background=(238, 238, 238)) -> np.ndarray:
    """Decode by file signature. Keep stored dimensions, including EXIF orientation."""
    try:
        if isinstance(source, np.ndarray):
            if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
                raise ValueError("Arrays must be uint8 RGB images.")
            if min(source.shape[:2]) < 2 or source.shape[0] * source.shape[1] > MAX_PIXELS:
                raise ValueError("Image dimensions must be at least 2 × 2 and at most 40 megapixels.")
            return np.ascontiguousarray(source)
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("Image file exceeds 80 MB.")
            data = path.read_bytes()
        elif isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        else:
            data = source.getvalue() if hasattr(source, "getvalue") else source.read()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("Image file exceeds 80 MB.")
        with Image.open(BytesIO(data)) as image:
            if image.format not in SUPPORTED_FORMATS:
                raise ValueError("Use PNG, JPG/JPEG, WebP, or BMP.")
            w, h = image.size
            if min(w, h) < 2 or w * h > MAX_PIXELS:
                raise ValueError("Image dimensions must be at least 2 × 2 and at most 40 megapixels.")
            image.verify()
        with Image.open(BytesIO(data)) as image:
            rgba = image.convert("RGBA")
            base = Image.new("RGBA", rgba.size, (*background, 255))
            return np.array(Image.alpha_composite(base, rgba).convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("The image is missing, corrupted, or unsupported.") from exc


@dataclass
class PreparedImage:
    rgb: np.ndarray
    gray: np.ndarray
    edges: np.ndarray
    binary: np.ndarray
    normalized: np.ndarray
    scaled: dict = field(default_factory=dict, repr=False)


def scaled_reference(template, width, height):
    """Reuse resized RGB/gray/edge arrays between methods within a variant search."""
    if (width, height) == template.gray.shape[::-1]:
        return template.rgb, template.gray, template.edges
    key = (width, height)
    if key not in template.scaled:
        rgb = cv2.resize(template.rgb, key, interpolation=cv2.INTER_AREA if width < template.rgb.shape[1] else cv2.INTER_CUBIC)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        value = (rgb, gray, cv2.Canny(gray, 45, 130))
        if width * height > 500_000:
            return value
        if len(template.scaled) >= 16:
            template.scaled.pop(next(iter(template.scaled)))
        template.scaled[key] = value
    return template.scaled[key]


def preprocess(rgb: np.ndarray) -> PreparedImage:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    normalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(gray, 45, 130)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)
    return PreparedImage(rgb, gray, edges, binary, normalized)


@lru_cache(maxsize=128)
def _cached_template(path: str, mtime_ns: int, size: int, theme: str):
    background = (24, 24, 28) if theme == "dark" else (238, 238, 238)
    return preprocess(load_rgb(path, background))


def load_template(path: Path, theme="auto") -> PreparedImage:
    stat = path.stat()
    return _cached_template(str(path.resolve()), stat.st_mtime_ns, stat.st_size, theme)
