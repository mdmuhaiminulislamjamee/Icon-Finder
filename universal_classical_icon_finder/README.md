# Universal Classical Icon Finder

An offline Python application that searches an entire screenshot for supplied reference icons, using OpenCV, NumPy, and Pillow. It has a Streamlit interface, command-line interface, portable platform profiles, and JSON/CSV/image exports. The detector uses no machine learning, OCR, neural networks, embeddings, models, or AI APIs.

## Your requested reference set

The starter library includes **20 identities and 42 PNG variants**:

| Reference group | Included controls |
| --- | --- |
| Frame 1 — Full screen | Diagonal arrows, four arrows, four corners, window |
| Frame 2 — Volume | Volume, muted volume |
| Frame 3 — Play & pause | Play, pause |
| Frame 4 — Navigation | Previous, next, fast forward, back 10 seconds, forward 10 seconds |
| Frame 5 — Reactions | Like, dislike, heart/favorite |
| Play bars 1–4 | White/gray, red, blue, cyan, including their progress tracks and thumbs |

Each identity has light and dark variants. Like and favorite also have active variants. See [the reference-library preview](sample_data/reference_library.png).

**Asset provenance:** the supplied picture was visible in chat but was not available as a local image file. The bundled icons are manually drawn approximations of those visible shapes and colors, not pixel-exact crops. Frames 26 and 27 are outside the requested set. Exact reference crops can replace the starter assets through **Reference library → Add an icon or visual variant**, followed by removal of the older variants. Templates retain a small margin around the glyph; result boxes include that margin.

The YouTube and Netflix profiles reuse subsets of the starter artwork to demonstrate profile separation. They are not verified captures of current YouTube or Netflix controls. Supply captures from the actual interface before judging accuracy on those platforms.

## Launch in this workspace

The dependencies are already installed in the workspace's parent `.venv`. Double-click **Start Icon Finder.cmd**, or run from this directory:

```powershell
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://127.0.0.1:8501**. Select an example screenshot and click **Find icons** to try it. All processing and uploads stay on your computer. The server binds to loopback and Streamlit usage telemetry is disabled in `.streamlit/config.toml`.

## Install on another computer

Python 3.10 or newer is required; validated here on Python 3.12.10 / Windows. The core and UI source use portable Python APIs, but macOS/Linux execution has not been tested in this workspace.

Windows PowerShell, from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Installation downloads Python packages. No internet connection is needed afterward. To install on a machine without internet access, download compatible wheels on a connected machine with the same operating system, architecture, and Python version:

```sh
python -m pip download -r requirements.txt --dest wheelhouse
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
```

`requirements-tested.txt` records the exact dependency versions used for validation. `requirements.txt` allows compatible upgrades. Streamlit has its own web/UI dependencies; the detector itself imports only OpenCV, NumPy, Pillow, and the standard library.

## Use the application

1. Upload a screenshot, or choose Desktop / Mobile sample data.
2. Choose **Unknown / Generic Platform** or another profile. Select the identities you want to search. Optional one-off references are accepted under **Add references for this search**.
3. Choose methods, minimum match score, theme, and scale range. References are matched at their stored size times the chosen scale. The native starter icons are 24 × 24 pixels and bars are 168 × 24; use scales around **0.5** for controls near 12 pixels tall.
4. Click **Find icons**. Inspect the annotated screenshot, exact coordinates, scores, method names, and timing table.
5. Download JSON, CSV, or the annotated PNG. Changing inputs hides stale results until another search runs.

Use tight reference crops with a few pixels of background. Include the background or use a transparent PNG, which is composited against the chosen theme. Auto mode uses light compositing for transparent references and allows inverted grayscale matching; choose Dark explicitly for transparent references on dark screenshots.

Start with template and edge matching. Contour matching is useful for connected silhouettes. SIFT/ORB/AKAZE are available under Features, but very small/simple controls often have too few keypoints. Feature matching is disabled by default because it adds work and may not help these icons. Lowering the threshold increases both sensitivity and false matches.

The optional priority region only changes result ordering. The full screenshot is always searched. There are no platform-specific coordinates, crops, or layout rules in the detector.

## Profiles and icon management

**Manage profiles** supports empty profiles, clones, ZIP import/export, and copying an identity with its variants between profiles. Built-in profiles are editable; create a clone to keep your preferred baseline.

**Reference library** supports uploads, aliases, categories, renaming, removal, and light/dark/hover/active/disabled variants. File dimensions are stored as resolution metadata. Uploaded files sharing an identity become additional variants. Removing an identity/variant edits the registry and retains the file on disk in case another identity references it. Profile exports include only registered files.

Profiles live in `platform_profiles/<name>/profile.json`, with images under `icons/`. Imports cannot overwrite an existing profile, use absolute paths, escape `icons/`, or exceed 80 MB expanded / 1000 archive entries. Image decoding is validated before imported profiles are saved.

Example profile:

```json
{
  "name": "my_app",
  "description": "My own controls",
  "icons": [
    {
      "name": "play",
      "aliases": ["start video"],
      "category": "transport",
      "templates": ["icons/play_light.png", "icons/play_dark.png"],
      "variant_metadata": {
        "icons/play_light.png": {"theme": "light", "state": "default", "resolution": [24, 24]},
        "icons/play_dark.png": {"theme": "dark", "state": "hover", "resolution": [24, 24]}
      },
      "enabled_methods": ["template_matching", "edge_matching", "shape_matching", "feature_matching"],
      "color_sensitive": false,
      "default_threshold": 0.86
    }
  ]
}
```

The selected global threshold overrides profile defaults in the UI/CLI. `color_sensitive` enables stricter color checking; it is on for the four starter bars. Any profile can use this property. The core has no checks for specific icon or platform names.

## Command line

From this directory, with the appropriate virtual-environment Python:

```sh
python cli.py sample_data/desktop_light.png --profile generic --icons play pause volume favorite fullscreen_corners play_bar_02 play_bar_03 --output results
python cli.py sample_data/mobile_dark.png --profile generic --theme dark --scale-min 0.75 --scale-max 1.5 --output mobile_results
python cli.py screenshot.png --profile my_app --methods template_matching edge_matching --threshold 0.9 --output results
```

Each run writes `results.json`, `results.csv`, and `annotated.png`. [Example JSON](sample_data/example_output/results.json), [example CSV](sample_data/example_output/results.csv), and [annotated screenshot](sample_data/example_output/annotated.png) are included.

## Pipeline and score interpretation

- Decode PNG, JPEG, WebP, or BMP by signature; reject corrupt, unsupported, oversized, or invalid images. Stored pixel dimensions are preserved, including EXIF-oriented files (no automatic rotation). Limit: 40 megapixels / 80 MB per image.
- Reuse RGB, grayscale, Canny edges, Otsu masks, and CLAHE-normalized maps. Base reference preprocessing uses an LRU cache invalidated by file modification time and size. Resized references are shared between methods with a bounded, per-search cache. Processing is sequential to avoid excessive CPU use.
- Template matching searches every feasible scale and skips references larger than the screenshot. CCOEFF_NORMED is the default; CCORR_NORMED and SQDIFF_NORMED also require centered structural agreement to reject bright blank backgrounds. Auto theme permits contrast inversion.
- Edge matching uses normalized correlation of Canny maps. Shape matching combines Hu moments, aspect ratio, and contour occupancy. It is best suited to isolated connected shapes, and is weaker for disconnected or outlined patterns.
- Feature matching uses classical descriptors, a two-neighbor ratio test, RANSAC homography, and aligned appearance verification. It removes the fitted instance's keypoints and searches for further copies. Screenshots share one feature/contour extraction per run.
- Apply optional profile-driven color verification, method weights, and the threshold. Fuse overlapping candidates **within each identity** by IoU, retain the strongest box, and collect supporting method names. Separate copies and different identities are retained. Identical artwork registered under different names can therefore produce multiple identity labels.

Scores range from zero to one after normalization, color checks, and method weights. **They are similarities, not calibrated probabilities of correctness.** The default weights in `DetectionSettings` are template 1.00, edges 0.97, contours 0.94, features 0.98. You can change them through the Python API. Fusion does not inflate scores.

Raw scores retain method meaning: correlation for templates/edges, normalized squared difference for SQDIFF (lower is better), Hu distance for contours (lower is better), and aligned appearance agreement for features. `processing_time_ms` is the method-call time for that reference variant, shared by candidates from that call; it is not an additive per-box cost. The report provides independent stage totals and total elapsed wall time.

Bounding boxes use half-open **original screenshot pixels**, `[x1, x2)` and `[y1, y2)`. The center is the exact midpoint and can contain `.5`. Annotation and export never resize the source. A box describes the whole reference crop, including margins. Geometrically transformed feature results use an axis-aligned enclosure of the projected template corners.

Peak extraction is bounded to 100 candidates per variant/scale (configurable through the Python API); feature matching fits up to 30 instances per variant. Extremely repetitive inputs can exceed these limits. Use a narrower selection/range to reduce latency and memory use on large screenshots.

## Python API

Run scripts with this directory on Python's module path:

```python
from config import DetectionSettings
from detector import detect
from icon_manager import ProfileManager

profile = ProfileManager().load("generic")
report = detect(
    "screenshot.png", profile, ["play", "pause"],
    DetectionSettings(threshold=0.9, scale_min=0.5, scale_max=2.0, scale_step=0.125),
)
print(report.results)
print(report.timing_ms)
```

## Validation

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite covers supported/invalid files, preprocessing/cache reuse, all normalized matching modes, repeated and scaled icons, theme inversion, slight color shifts, partial occlusion, no-match inputs, all four play-bar colors, contour matching, feature homography with repeated textured references, duplicate filtering, original coordinates, 1920 × 1080 inputs, region priority, new/unknown/custom platforms, profile isolation and ZIP round trips, malicious archive paths, exports, and a Streamlit demo workflow.

Two synthetic screenshots have explicit ground truth. The desktop sample contains eight controls and the mobile sample six. These are exact/reconstructed-template sanity checks, **not an accuracy benchmark on real platforms**. See [VALIDATION.md](VALIDATION.md) for the verified run.

## Limits

Classical matching cannot infer an icon's meaning or recognize an arbitrary redesign. Real screenshot accuracy depends on reference fidelity, crop, size, theme, threshold, compression, and surrounding content. Motion blur, extremely small controls, complex backgrounds, strong compression, major color changes, severe occlusion, perspective distortion, animation, and platform UI changes can reduce recall or create false positives. Feature homography offers limited perspective tolerance when the reference has sufficient detail.

Partial occlusion is tested for graceful behavior and reduced similarity; recovery is not guaranteed. Monochrome shapes can match similar shapes elsewhere. Progress bars are visual templates: different lengths, aspect ratios, or thumb positions need more reference variants; uniform scale search does not independently stretch a track or move its thumb. Header text and green labels in the reference image are not detector inputs. No overall accuracy percentage is claimed.

## Files

```text
app.py                       Streamlit interface
cli.py                       Command-line entry point
config.py                    Validated settings and method weights
detector/                    Preprocessing, four matchers, ranking, fusion, coordinates, exports
icon_manager/                Profiles, registry, variants, portable ZIPs
platform_profiles/           generic, youtube, netflix, custom_demo
sample_data/                 Reconstructed library, synthetic screenshots, ground truth, sample exports
tests/                       Unit, integration, and Streamlit tests
tools/generate_starter_assets.py  Reproducible starter artwork and samples
```

The asset-generation script is for regeneration during development and can overwrite bundled profiles. It is never needed for ordinary app use.
