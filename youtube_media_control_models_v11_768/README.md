# YouTube Media-Control Models V11 — 768 px handover

## Delivery summary

This package follows the original V1 two-model format:

1. `youtube_controls_progress` detects 12 media-control classes and the complete
   `progress_bar`.
2. `youtube_playhead` is a one-class specialist that detects `playhead`.

Only ONNX and TFLite deployment models are shipped. There are no PyTorch or
SavedModel files in the handover.

## Recommended files

Use Float32 first for acceptance testing:

- `models/tflite/youtube_controls_progress_float32.tflite`
- `models/tflite/youtube_playhead_float32.tflite`

Smaller Float16-weight TFLite alternatives are also included. Their input and
output tensors remain Float32. Fixed-shape Float32 ONNX alternatives are under
`models/onnx/`. Select the runtime supported by the application environment.

## Supported training scope

The frozen `current_v11` dataset contains YouTube screenshots from:

- desktop web;
- mobile web;
- tablet web;
- native Android phone;
- native Android tablet;
- native Android foldable;
- regular/full-video and Shorts layouts.

This is a YouTube model. The examples show different operating environments,
not support for unrelated video services.

## Required runtime pipeline

For every source frame:

1. Use the complete supplied screenshot/frame; no separate browser, player, or
   UI-region crop was used by the delivered pipeline.
2. Convert to RGB.
3. Preserve aspect ratio and letterbox to exactly `768 x 768` with RGB
   `(114, 114, 114)` padding. Do not stretch the screenshot.
4. Convert pixels to Float32 and divide by `255`.
5. Run both models.
6. Decode raw YOLO `xywh` boxes and class scores.
7. Apply confidence filtering and class-aware NMS.
8. Remove padding and divide by the saved resize gain to restore source-frame
   coordinates.
9. Use `progress_bar` only from the controls model and `playhead` only from the
   playhead model.
10. Associate at most one playhead with the detected active progress bar.
11. Optionally apply temporal smoothing after detections are restored to source-frame
    coordinates.

Do not resize to a hard-coded phone or desktop resolution. A 720p, 1080p,
portrait, landscape, square, tablet, or foldable screenshot all use the same
aspect-ratio-preserving 768 letterbox procedure.

See `HANDOVER.md` for the mandatory implementation contract: tensor layouts,
class indices, decoding, coordinate restoration, temporal smoothing,
and integration checks.

## Classes

Controls/progress model:

| Index | Class |
|---:|---|
| 0 | `play` |
| 1 | `pause` |
| 2 | `volume_on` |
| 3 | `volume_muted` |
| 4 | `fullscreen` |
| 5 | `exit_fullscreen` |
| 6 | `previous` |
| 7 | `next` |
| 8 | `like` |
| 9 | `dislike` |
| 10 | `seek_backward` |
| 11 | `seek_forward` |
| 12 | `progress_bar` |

Playhead model: index `0` is `playhead`.

## Reference inference

From the project root:

```bash
conda run --no-capture-output -n mediacontroldetector \
  python output/youtube_media_control_models_v11_768/scripts/infer.py \
  /absolute/path/to/screenshot.png \
  --backend tflite --precision float32 \
  --save /tmp/detection.jpg --json /tmp/detection.json
```

ONNX:

```bash
conda run --no-capture-output -n mediacontroldetector \
  python output/youtube_media_control_models_v11_768/scripts/infer.py \
  /absolute/path/to/screenshot.png \
  --backend onnx --save /tmp/detection.jpg
```

The script is a readable integration reference. It does not synthesize or
recover a progress bar when the model does not emit one.

## Device examples

`youtube_tflite_visuals/` contains direct Float32 TFLite examples covering the major
web and native-device environments. They are supplied as integration smoke
inputs, not as an application accuracy report.

Use `youtube_tflite_visuals/android_devices_contact_sheet.jpg` for a larger view of
the phone portrait/Shorts, phone landscape/full-video, tablet and foldable
examples. The images are raw screenshots, so they do not include an emulator
window or device skin.

## Package contents

- `AGENTS.md` — mandatory rules automatically visible to an app-team Codex.
- `models/onnx/` — two fixed `768 x 768` Float32 ONNX models.
- `models/tflite/` — Float32 and Float16-weight TFLite variants.
- `scripts/infer.py` — reference preprocessing, decoding and postprocessing.
- `youtube_tflite_visuals/` — numbered-box TFLite examples using the V1 side-panel format.
- `HANDOVER.md` — complete implementation contract and prohibitions.
- `manifest.json` and `model_contracts.json` — machine-readable delivery data.
