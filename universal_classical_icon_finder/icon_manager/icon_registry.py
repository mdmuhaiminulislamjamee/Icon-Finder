from copy import deepcopy
from io import BytesIO
from PIL import Image
from .profile_manager import slug, safe_template_path
from detector.preprocessing import load_rgb


class IconRegistry:
    def __init__(self, manager):
        self.manager = manager

    def add(self, profile_name, icon_name, image_data, category="custom", aliases=(), theme="all", state="default", color_sensitive=False):
        profile = self.manager.load(profile_name)
        data = deepcopy(profile.metadata)
        name = slug(icon_name)
        rgb = load_rgb(image_data)
        # Keep alpha for custom PNGs; the detector composites it to the chosen theme.
        image = Image.open(BytesIO(image_data)).convert("RGBA")
        existing = next((i for i in data["icons"] if i["name"] == name), None)
        if existing is None:
            if len(data["icons"]) >= 300:
                raise ValueError("This profile already has 300 icon identities.")
            existing = {"name": name, "aliases": list(aliases), "category": category,
                        "templates": [], "variant_metadata": {}, "default_threshold": 0.86,
                        "color_sensitive": bool(color_sensitive),
                        "enabled_methods": ["template_matching", "edge_matching", "shape_matching", "feature_matching"]}
            data["icons"].append(existing)
        elif len(existing["templates"]) >= 100:
            raise ValueError("This icon already has 100 variants.")
        serial = len(existing["templates"]) + 1
        relative = f"icons/{name}_{serial}.png"
        while safe_template_path(profile.directory, relative).exists():
            serial += 1
            relative = f"icons/{name}_{serial}.png"
        destination = safe_template_path(profile.directory, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination)
        existing["templates"].append(relative)
        existing.setdefault("variant_metadata", {})[relative] = {
            "theme": theme, "state": state, "resolution": list(rgb.shape[1::-1])}
        self.manager.save(data)
        return name

    def update(self, profile_name, old_name, name, aliases, category, color_sensitive=None):
        data = deepcopy(self.manager.load(profile_name).metadata)
        new_name = slug(name)
        if old_name != new_name and any(i["name"] == new_name for i in data["icons"]):
            raise ValueError("An icon with that name already exists.")
        icon = next(i for i in data["icons"] if i["name"] == old_name)
        icon.update(name=new_name, aliases=list(aliases), category=category)
        if color_sensitive is not None:
            icon["color_sensitive"] = bool(color_sensitive)
        self.manager.save(data)

    def remove(self, profile_name, name):
        data = deepcopy(self.manager.load(profile_name).metadata)
        data["icons"] = [i for i in data["icons"] if i["name"] != name]
        # Preserve files in case another identity shares a template.
        self.manager.save(data)

    def copy(self, source_name, target_name, icon_name):
        source = self.manager.load(source_name)
        target = self.manager.load(target_name)
        icon = next(i for i in source.icons if i["name"] == icon_name)
        if any(i["name"] == icon_name for i in target.icons):
            raise ValueError("Target profile already has this icon. Upload a variant instead.")
        data = deepcopy(target.metadata)
        copied = deepcopy(icon)
        templates = []
        metadata = {}
        for index, relative in enumerate(icon["templates"]):
            new_relative = f"icons/{icon_name}_shared_{index}.png"
            serial = 0
            while safe_template_path(target.directory, new_relative).exists():
                serial += 1
                new_relative = f"icons/{icon_name}_shared_{index}_{serial}.png"
            safe_template_path(target.directory, new_relative).write_bytes(safe_template_path(source.directory, relative).read_bytes())
            templates.append(new_relative)
            metadata[new_relative] = icon.get("variant_metadata", {}).get(relative, {})
        copied.update(templates=templates, variant_metadata=metadata)
        data["icons"].append(copied)
        self.manager.save(data)
