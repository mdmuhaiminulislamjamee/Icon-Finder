from copy import deepcopy


def select_variants(icon, theme):
    for relative in icon["templates"]:
        metadata = icon.get("variant_metadata", {}).get(relative, {})
        if theme == "auto" or metadata.get("theme", "all") in {"all", theme}:
            yield relative


def remove_variant(manager, profile_name, icon_name, relative):
    data = deepcopy(manager.load(profile_name).metadata)
    icon = next(i for i in data["icons"] if i["name"] == icon_name)
    if len(icon["templates"]) <= 1:
        raise ValueError("Keep at least one variant, or remove the icon.")
    if relative not in icon["templates"]:
        raise ValueError("Unknown variant.")
    icon["templates"].remove(relative)
    icon.get("variant_metadata", {}).pop(relative, None)
    manager.save(data)
