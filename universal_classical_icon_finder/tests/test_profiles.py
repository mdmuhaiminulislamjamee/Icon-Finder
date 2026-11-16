from io import BytesIO
import json
import zipfile
from copy import deepcopy
import pytest
from icon_manager import ProfileManager
from icon_manager.icon_registry import IconRegistry
from icon_manager.variant_manager import remove_variant
from detector import detect
from tools.generate_starter_assets import draw_icon
from conftest import scene


def png(name="play"):
    buffer = BytesIO()
    draw_icon(name).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("profile_name", ["generic", "youtube", "netflix", "custom_demo"])
def test_all_shipped_profiles_use_same_detector(manager, settings, profile_name):
    report = detect(scene([("play", 120, 80, 1)]), manager.load(profile_name), ["play"], settings)
    assert len(report.results) == 1
    assert report.results[0]["platform"] == profile_name
    assert report.results[0]["bounding_box"]["x1"] == 120


def test_new_unknown_platform_and_export_roundtrip(tmp_path, settings):
    manager = ProfileManager(tmp_path)
    manager.create("never_seen_before")
    registry = IconRegistry(manager)
    registry.add("never_seen_before", "start", png(), aliases=["play"])
    registry.add("never_seen_before", "start", png(), theme="dark", state="hover")
    manager.import_zip(manager.export_zip("never_seen_before"), "round_trip")
    assert len(manager.load("round_trip").icons[0]["templates"]) == 2
    report = detect(scene([("play", 31, 79, 1)]), manager.load("round_trip"), ["play"], settings)
    assert len(report.results) == 1 and report.results[0]["icon"] == "start"
    registry.update("round_trip", "start", "begin", ["start playback"], "transport")
    assert manager.load("round_trip").icons[0]["category"] == "transport"
    remove_variant(manager, "round_trip", "begin", manager.load("round_trip").icons[0]["templates"][1])
    with pytest.raises(ValueError, match="at least one"):
        remove_variant(manager, "round_trip", "begin", manager.load("round_trip").icons[0]["templates"][0])
    registry.remove("round_trip", "begin")
    assert manager.load("round_trip").icons == []


def test_platform_mutations_do_not_change_generic(tmp_path, manager, settings):
    local = ProfileManager(tmp_path)
    local.import_zip(manager.export_zip("generic"))
    before = local.export_zip("generic")
    local.create("specific", clone_from="generic")
    IconRegistry(local).update("specific", "play", "custom_play", [], "custom")
    assert local.export_zip("generic") == before
    assert detect(scene([("play", 70, 20, 1)]), local.load("generic"), ["play"], settings).results[0]["icon"] == "play"


def test_copy_and_invalid_icon(tmp_path):
    manager = ProfileManager(tmp_path)
    manager.create("one")
    manager.create("two")
    registry = IconRegistry(manager)
    registry.add("one", "play", png())
    registry.copy("one", "two", "play")
    assert manager.load("two").icons[0]["name"] == "play"
    with pytest.raises(ValueError):
        registry.add("one", "invalid", b"corrupted")
    with pytest.raises(ValueError):
        manager.create("one")


def test_imported_empty_profile_accepts_new_references(tmp_path, settings):
    manager = ProfileManager(tmp_path)
    manager.create("empty")
    manager.import_zip(manager.export_zip("empty"), "imported_empty")
    IconRegistry(manager).add("imported_empty", "play", png(), color_sensitive=True)
    assert manager.load("imported_empty").icons[0]["color_sensitive"] is True
    report = detect(scene([("play", 45, 75, 1)]), manager.load("imported_empty"), settings=settings)
    assert len(report.results) == 1


@pytest.mark.parametrize("path", ["../escape.png", "icons/../../escape.png", "C:/escape.png", "icons\\escape.png"])
def test_archive_path_traversal_rejected(tmp_path, path):
    data = {"name": "bad", "icons": [{"name": "play", "templates": [path]}]}
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("profile.json", json.dumps(data))
        archive.writestr(path, png())
    with pytest.raises(ValueError):
        ProfileManager(tmp_path).import_zip(buffer.getvalue())
    assert not (tmp_path / "bad").exists()


def test_corrupt_zip_and_missing_templates_rejected(tmp_path):
    manager = ProfileManager(tmp_path)
    with pytest.raises(ValueError):
        manager.import_zip(b"not a zip")
    manager.create("empty")
    data = deepcopy(manager.load("empty").metadata)
    data["icons"] = [{"name": "play", "templates": ["icons/missing.png"]}]
    with pytest.raises(ValueError, match="Missing"):
        manager.save(data)
