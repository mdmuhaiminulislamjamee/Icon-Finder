"""Hand-authored reconstructions of the user's chat reference; no models or downloads.

The chat image was not provided as a local bitmap. These are replaceable visual
approximations, not pixel-exact crops. Only this sample generator uses layouts.
"""
from pathlib import Path
import json
import sys
import math
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import METHODS

GROUPS = {
    "Frame 1 · Full screen": ["fullscreen_diagonal", "fullscreen_arrows", "fullscreen_corners", "fullscreen_window"],
    "Frame 2 · Volume": ["volume", "volume_muted"],
    "Frame 3 · Play & pause": ["play", "pause"],
    "Frame 4 · Navigation": ["previous", "next", "fast_forward", "skip_backward_10", "skip_forward_10"],
    "Frame 5 · Reactions": ["like", "dislike", "favorite"],
    "Play bars 1–4": [f"play_bar_{i:02}" for i in range(1, 5)],
}
ALIASES = {"play": ["play_button", "start video"], "pause": ["pause_button"],
           "volume_muted": ["mute", "muted"], "favorite": ["heart"],
           "like": ["thumbs up"], "dislike": ["thumbs down"]}


def draw_icon(name, theme="light", state="default"):
    factor = 4
    bg = (24, 24, 28) if theme == "dark" else (238, 238, 238)
    fg = (226, 226, 235) if theme == "dark" else (30, 33, 53)
    if state == "active":
        fg = (27, 116, 246)
    if state == "disabled":
        fg = tuple(round(a * 0.4 + b * 0.6) for a, b in zip(fg, bg))
    if name.startswith("play_bar"):
        image = Image.new("RGB", (168 * factor, 24 * factor), bg)
        draw = ImageDraw.Draw(image)
        style = int(name[-2:])
        played = {1: (255, 255, 255), 2: (255, 0, 65), 3: (23, 116, 255), 4: (0, 220, 238)}[style]
        rest = {1: (126, 126, 126), 2: (80, 70, 47), 3: (206, 214, 224), 4: (122, 131, 133)}[style]
        if theme == "dark" and style in (1, 3):
            rest = (78, 85, 99)
        progress = 57 if style != 4 else 56
        draw.line((2*factor, 12*factor, 166*factor, 12*factor), fill=rest, width=2*factor)
        draw.line((2*factor, 12*factor, progress*factor, 12*factor), fill=played, width=2*factor)
        radius = 6 if style in (1, 2) else 4
        thumb = (255, 255, 255) if style == 4 else played
        draw.ellipse(((progress-radius)*factor, (12-radius)*factor, (progress+radius)*factor, (12+radius)*factor), fill=thumb)
        return image.resize((168, 24), Image.Resampling.LANCZOS)
    image = Image.new("RGB", (24*factor, 24*factor), bg)
    d = ImageDraw.Draw(image)
    def line(points, width=1.15):
        d.line([(round(x*factor), round(y*factor)) for x, y in points], fill=fg, width=max(1, round(width*factor)), joint="curve")
    def rect(box, fill=None, radius=0):
        d.rounded_rectangle(tuple(round(v*factor) for v in box), radius=round(radius*factor),
                            outline=fg, fill=fill, width=round(1.15*factor))
    def arc(box, start, end):
        d.arc(tuple(round(v*factor) for v in box), start, end, fill=fg, width=round(1.15*factor))
    def polygon(points, filled=False):
        points = [(round(x*factor), round(y*factor)) for x, y in points]
        d.polygon(points, fill=fg if filled else None)
        d.line(points + [points[0]], fill=fg, width=round(1.15*factor), joint="curve")
    if name == "play":
        polygon([(7, 5), (18, 12), (7, 19)])
    elif name == "pause":
        rect((6, 5, 9, 19), radius=0.6)
        rect((15, 5, 18, 19), radius=0.6)
    elif name in {"next", "previous"}:
        if name == "next":
            polygon([(6, 6), (15, 12), (6, 18)])
            line([(18, 6), (18, 18)])
        else:
            polygon([(18, 6), (9, 12), (18, 18)])
            line([(6, 6), (6, 18)])
    elif name == "fast_forward":
        polygon([(4, 6), (12, 12), (4, 18)])
        polygon([(12, 6), (20, 12), (12, 18)])
    elif name.startswith("skip_"):
        arc((4, 4, 20, 20), -80, 245)
        if name == "skip_backward_10":
            line([(8, 3), (4, 6), (8, 8)])
        else:
            image = ImageOps.mirror(image)
            d = ImageDraw.Draw(image)
        d.text((7*factor, 7*factor), "10", fill=fg, font=ImageFont.load_default(size=7*factor))
    elif name in {"volume", "volume_muted"}:
        polygon([(4, 9), (8, 9), (13, 5), (13, 19), (8, 15), (4, 15)])
        if name == "volume":
            arc((9, 7, 19, 17), -55, 55)
            arc((8, 4, 23, 20), -48, 48)
        else:
            line([(17, 9), (22, 15)])
            line([(22, 9), (17, 15)])
    elif name in {"like", "dislike"}:
        polygon([(9, 10), (12, 5), (12, 3), (15, 4), (15, 9), (20, 9), (21, 11),
                 (18, 19), (9, 19)])
        rect((4, 10, 8, 20), radius=0.4)
        if name == "dislike":
            image = ImageOps.flip(image)
    elif name == "favorite":
        points = []
        for step in range(121):
            t = 2 * math.pi * step / 120
            x = 12 + 0.52 * 16 * math.sin(t) ** 3
            y = 11 - 0.52 * (13*math.cos(t) - 5*math.cos(2*t) - 2*math.cos(3*t) - math.cos(4*t))
            points.append((x, y))
        polygon(points, filled=state == "active")
    elif name == "fullscreen_diagonal":
        line([(5, 5), (19, 19)])
        line([(5, 10), (5, 5), (10, 5)])
        line([(14, 19), (19, 19), (19, 14)])
    elif name == "fullscreen_arrows":
        for x, y, dx, dy in [(5, 5, 1, 1), (19, 5, -1, 1), (5, 19, 1, -1), (19, 19, -1, -1)]:
            line([(x+dx*5, y+dy*5), (x, y)])
            line([(x+dx*4, y), (x, y), (x, y+dy*4)])
    elif name == "fullscreen_corners":
        for x, y, dx, dy in [(5, 5, 1, 1), (19, 5, -1, 1), (5, 19, 1, -1), (19, 19, -1, -1)]:
            line([(x+dx*4, y), (x, y), (x, y+dy*4)])
    elif name == "fullscreen_window":
        rect((5, 5, 19, 19), radius=3)
        line([(12, 5), (12, 8)], width=2)
        line([(5, 12), (8, 12)], width=2)
        rect((13, 13, 21, 21), fill=bg, radius=1)
    else:
        raise ValueError(name)
    return image.resize((24, 24), Image.Resampling.LANCZOS)


def build_profiles():
    directory = ROOT / "platform_profiles" / "generic"
    (directory / "icons").mkdir(parents=True, exist_ok=True)
    profile = {"name": "generic", "description": "Unknown / Generic Platform · Media controls from Frames 1–5 and Play bars 1–4",
               "asset_provenance": "Hand-authored approximations of a chat-preview reference; not pixel-exact crops or official platform assets.", "icons": []}
    for category, names in GROUPS.items():
        for name in names:
            icon = {"name": name, "aliases": ALIASES.get(name, []), "category": category,
                    "templates": [], "variant_metadata": {}, "enabled_methods": list(METHODS), "default_threshold": 0.86,
                    "color_sensitive": name.startswith("play_bar")}
            for theme in ("light", "dark"):
                relative = f"icons/{name}_{theme}.png"
                draw_icon(name, theme).save(directory / relative)
                icon["templates"].append(relative)
                icon["variant_metadata"][relative] = {"theme": theme, "state": "default", "resolution": list(draw_icon(name).size)}
            if name in {"favorite", "like"}:
                relative = f"icons/{name}_active.png"
                draw_icon(name, "light", "active").save(directory / relative)
                icon["templates"].append(relative)
                icon["variant_metadata"][relative] = {"theme": "light", "state": "active"}
            profile["icons"].append(icon)
    (directory / "profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    subsets = {"youtube": {"play", "pause", "volume", "volume_muted", "fullscreen_corners", "like", "dislike", "play_bar_02"},
               "netflix": {"play", "pause", "volume", "volume_muted", "fullscreen_corners", "skip_backward_10", "skip_forward_10", "play_bar_02"},
               "custom_demo": {"play", "pause", "favorite", "play_bar_03"}}
    for name, subset in subsets.items():
        target = ROOT / "platform_profiles" / name
        (target / "icons").mkdir(parents=True, exist_ok=True)
        data = deepcopy(profile)
        data.update(name=name, description=f"{'Custom' if name == 'custom_demo' else name.title()} example profile using the shared starter artwork; replace with actual platform captures.")
        data["icons"] = [i for i in data["icons"] if i["name"] in subset]
        for icon in data["icons"]:
            for relative in icon["templates"]:
                (target / relative).write_bytes((directory / relative).read_bytes())
        (target / "profile.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_board():
    image = Image.new("RGB", (1200, 580), "#eeeeee")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=25)
    label = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=11)
    draw.text((32, 24), "REFERENCE ICON LIBRARY", fill="#202135", font=title)
    draw.text((32, 62), "Frames 1-5 + Play bars 1-4 · Recreated from the supplied preview", fill="#687079", font=label)
    positions = [(32, 110, 345), (410, 110, 210), (658, 110, 230), (32, 280, 530), (610, 280, 300)]
    for (category, names), (x, y, width) in zip(list(GROUPS.items())[:5], positions):
        draw.rectangle((x, y, x+width, y+28), fill="#b9ee00")
        draw.text((x+10, y+6), category, fill="#202135", font=label)
        for index, name in enumerate(names):
            image.paste(draw_icon(name).resize((36, 36), Image.Resampling.NEAREST), (x+index*90+12, y+44))
            caption = name.replace("fullscreen_", "").replace("skip_", "").replace("volume_", "")
            draw.text((x+index*90, y+86), caption, fill="#545a64", font=small)
    for index in range(4):
        x, y = 32+index*288, 450
        draw.rectangle((x, y, x+240, y+28), fill="#b9ee00")
        draw.text((x+12, y+6), f"Play bar {index+1:02}", fill="#202135", font=label)
        image.paste(draw_icon(f"play_bar_{index+1:02}"), (x+12, y+48))
    image.save(ROOT / "sample_data" / "reference_library.png")


def build_samples():
    target = ROOT / "sample_data"
    target.mkdir(exist_ok=True)
    for name, theme, size, placements in [
        ("desktop_light", "light", (1280, 720), [("play", 90, 170, 1), ("play", 780, 422, 1.5),
            ("pause", 470, 260, 1), ("volume", 950, 566, 1), ("favorite", 1060, 130, 1),
            ("fullscreen_corners", 1120, 566, 1), ("play_bar_02", 120, 575, 1), ("play_bar_03", 760, 575, 1)]),
        ("mobile_dark", "dark", (480, 850), [("play", 87, 250, 1.5), ("pause", 320, 580, 1),
            ("volume_muted", 62, 690, 1), ("favorite", 366, 96, 1), ("play_bar_04", 135, 705, 1),
            ("fullscreen_corners", 382, 580, 1)]),
    ]:
        bg = (238, 238, 238) if theme == "light" else (24, 24, 28)
        image = Image.new("RGB", size, bg)
        draw = ImageDraw.Draw(image)
        color = "#303442" if theme == "light" else "#d8dbe3"
        draw.text((40, 36), "MEDIA / SAMPLE WORKSPACE", fill=color, font=ImageFont.load_default(size=20))
        draw.text((40, 72), "Synthetic validation image · controls may appear anywhere", fill=color,
                  font=ImageFont.load_default(size=12))
        draw.rounded_rectangle((40, 120, size[0]-40, 360 if theme == "light" else 480), radius=12,
                               outline="#bdc2ca" if theme == "light" else "#3d414e", width=1)
        truth = []
        for identity, x, y, scale in placements:
            icon = draw_icon(identity, theme)
            if scale != 1:
                icon = icon.resize((round(icon.width*scale), round(icon.height*scale)), Image.Resampling.BICUBIC)
            image.paste(icon, (x, y))
            truth.append({"icon": identity, "bounding_box": {"x1": x, "y1": y, "x2": x+icon.width, "y2": y+icon.height}})
        image.save(target / f"{name}.png")
        (target / f"{name}_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    build_board()


if __name__ == "__main__":
    build_profiles()
    build_samples()
    print("Created 20 icon identities, 42 template variants, four profiles, and sample screenshots.")
