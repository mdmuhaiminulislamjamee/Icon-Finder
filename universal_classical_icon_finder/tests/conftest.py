from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DetectionSettings
from icon_manager import ProfileManager
from tools.generate_starter_assets import draw_icon


@pytest.fixture
def manager():
    return ProfileManager(ROOT / "platform_profiles")


@pytest.fixture
def settings():
    return DetectionSettings(methods=("template_matching", "edge_matching"), threshold=0.9,
                             scale_min=1, scale_max=1, scale_step=0.1)


def scene(placements, theme="light", size=(360, 240)):
    bg = (238, 238, 238) if theme == "light" else (24, 24, 28)
    image = np.full((size[1], size[0], 3), bg, dtype=np.uint8)
    for name, x, y, scale in placements:
        icon = draw_icon(name, theme)
        if scale != 1:
            from PIL import Image
            icon = icon.resize((round(icon.width*scale), round(icon.height*scale)), Image.Resampling.BICUBIC)
        image[y:y+icon.height, x:x+icon.width] = np.array(icon)
    return image
