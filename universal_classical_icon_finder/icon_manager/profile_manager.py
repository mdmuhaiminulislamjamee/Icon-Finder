"""Portable JSON profiles and bounded, traversal-safe ZIP import/export."""
from dataclasses import dataclass
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile

from config import METHODS, PROFILE_ROOT
from detector.preprocessing import load_rgb


def slug(value):
    value = re.sub(r"[^a-z0-9_-]+", "_", str(value).strip().lower()).strip("_")
    if not value or len(value) > 80:
        raise ValueError("Use a name containing 1–80 letters, numbers, underscores, or hyphens.")
    return value


def safe_template_path(directory, relative):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative:
        raise ValueError("Template paths must be relative POSIX paths.")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.parts[:1] != ("icons",):
        raise ValueError("Templates must stay inside the profile's icons directory.")
    result = (Path(directory) / relative).resolve()
    if not result.is_relative_to(Path(directory).resolve()):
        raise ValueError("Template path escapes the profile.")
    return result


@dataclass
class Profile:
    name: str
    description: str
    icons: list[dict]
    directory: Path
    metadata: dict


def validate_profile(data, directory, check_files=True):
    if not isinstance(data, dict) or not isinstance(data.get("name"), str):
        raise ValueError("Profile must be a JSON object with a name.")
    if slug(data["name"]) != data["name"]:
        raise ValueError("Profile name must use lowercase letters, numbers, underscores, or hyphens.")
    icons = data.get("icons")
    if not isinstance(icons, list) or len(icons) > 300:
        raise ValueError("Profile icons must be a list of at most 300 entries.")
    names = set()
    for icon in icons:
        if not isinstance(icon, dict) or not isinstance(icon.get("name"), str):
            raise ValueError("Every icon needs a name.")
        if slug(icon["name"]) != icon["name"] or icon["name"] in names:
            raise ValueError("Icon names must be unique, valid identifiers.")
        names.add(icon["name"])
        templates = icon.get("templates")
        if not isinstance(templates, list) or not templates or len(templates) > 100:
            raise ValueError("Every icon needs 1–100 reference templates.")
        for relative in templates:
            path = safe_template_path(directory, relative)
            if check_files and not path.is_file():
                raise ValueError(f"Missing reference template: {relative}")
        if not isinstance(icon.get("aliases", []), list) or not all(isinstance(a, str) for a in icon.get("aliases", [])):
            raise ValueError("Aliases must be a list of strings.")
        methods = icon.get("enabled_methods", list(METHODS))
        if not isinstance(methods, list) or not methods or not set(methods) <= set(METHODS):
            raise ValueError("Icon methods must be supported detection methods.")
        threshold = icon.get("default_threshold", 0.86)
        if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise ValueError("Icon thresholds must be between zero and one.")
        metadata = icon.get("variant_metadata", {})
        if not isinstance(metadata, dict) or not all(isinstance(v, dict) for v in metadata.values()):
            raise ValueError("Variant metadata must map template paths to objects.")
    return data


class ProfileManager:
    def __init__(self, root=PROFILE_ROOT):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, name):
        if slug(name) != name:
            raise ValueError("Invalid profile identifier.")
        directory = (self.root / name).resolve()
        if not directory.is_relative_to(self.root):
            raise ValueError("Profile escapes the profile root.")
        return directory

    def list_profiles(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and (p / "profile.json").is_file()
                      and not p.name.startswith("."))

    def load(self, name):
        directory = self.directory(name)
        try:
            data = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read profile {name}.") from exc
        validate_profile(data, directory)
        if data["name"] != name:
            raise ValueError("Profile identifier must match its directory.")
        return Profile(name, str(data.get("description", "")), data["icons"], directory, data)

    def save(self, data):
        directory = self.directory(data["name"])
        validate_profile(data, directory)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / ".profile.json.tmp"
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(directory / "profile.json")

    def create(self, name, description="", clone_from=None):
        name = slug(name)
        directory = self.directory(name)
        if directory.exists():
            raise ValueError("That profile already exists. Choose another name.")
        if clone_from:
            original = self.load(clone_from)
            shutil.copytree(original.directory, directory)
            data = deepcopy(original.metadata)
            data.update(name=name, description=description)
        else:
            (directory / "icons").mkdir(parents=True)
            data = {"name": name, "description": description, "icons": []}
        self.save(data)
        return self.load(name)

    def export_zip(self, name):
        profile = self.load(name)
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("profile.json", json.dumps(profile.metadata, indent=2))
            for relative in sorted({t for icon in profile.icons for t in icon["templates"]}):
                archive.write(safe_template_path(profile.directory, relative), relative)
        return buffer.getvalue()

    def import_zip(self, data, name=None):
        if len(data) > 80_000_000:
            raise ValueError("Profile archive exceeds 80 MB.")
        try:
            with zipfile.ZipFile(BytesIO(data)) as archive:
                entries = archive.infolist()
                if len(entries) > 1000 or sum(e.file_size for e in entries) > 80_000_000:
                    raise ValueError("Archive expands beyond the 80 MB / 1000 file limit.")
                entry_names = [e.filename for e in entries]
                if len(entry_names) != len({n.casefold() for n in entry_names}):
                    raise ValueError("Archive contains duplicate paths.")
                if "profile.json" not in entry_names or archive.getinfo("profile.json").file_size > 1_000_000:
                    raise ValueError("Archive must contain a profile.json at its root (at most 1 MB).")
                profile_data = json.loads(archive.read("profile.json"))
                profile_data["name"] = slug(name or profile_data.get("name", "imported"))
                target = self.directory(profile_data["name"])
                if target.exists():
                    raise ValueError("That profile already exists. Import under a new name.")
                with tempfile.TemporaryDirectory(prefix=".import_", dir=self.root) as staging:
                    staging = Path(staging)
                    (staging / "icons").mkdir()
                    validate_profile(profile_data, staging, check_files=False)
                    expected = {t for i in profile_data["icons"] for t in i["templates"]}
                    for entry in entries:
                        if entry.filename == "profile.json" or entry.is_dir():
                            continue
                        destination = safe_template_path(staging, entry.filename)
                        if entry.filename not in expected:
                            raise ValueError("Archive contains an unreferenced file.")
                        content = archive.read(entry)
                        load_rgb(content)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(content)
                    validate_profile(profile_data, staging)
                    (staging / "profile.json").write_text(json.dumps(profile_data, indent=2), encoding="utf-8")
                    shutil.copytree(staging, target)
                return self.load(profile_data["name"])
        except (zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, RuntimeError) as exc:
            raise ValueError("Invalid profile archive.") from exc
