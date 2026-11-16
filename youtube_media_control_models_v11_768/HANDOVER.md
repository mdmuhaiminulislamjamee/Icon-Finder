# YouTube Media-Control Detector V11 — handover

## Implementation task

Integrate the supplied two-model detector by following this document and the
reference implementation in `scripts/infer.py`.

Run both models on the same frame:

1. `youtube_controls_progress` detects media controls and `progress_bar`.
2. `youtube_playhead` detects `playhead`.

## Model files

TFLite:

```text
models/tflite/youtube_controls_progress_float32.tflite
models/tflite/youtube_controls_progress_float16.tflite
models/tflite/youtube_playhead_float32.tflite
models/tflite/youtube_playhead_float16.tflite
```

ONNX:

```text
models/onnx/youtube_controls_progress.onnx
models/onnx/youtube_playhead.onnx
```

Use a matching pair from one runtime. Start integration with the Float32
TFLite pair. The Float16 TFLite files use reduced-precision weights but retain
Float32 input and output tensors.

## Class mapping

Controls/progress model:

```text
0  play
1  pause
2  volume_on
3  volume_muted
4  fullscreen
5  exit_fullscreen
6  previous
7  next
8  like
9  dislike
10 seek_backward
11 seek_forward
12 progress_bar
```

Playhead model:

```text
0 playhead
```

`progress_bar` represents the complete visible seek track. `playhead`
represents the current position marker.

## Processing pipeline

For every frame:

```text
frame
  -> convert to RGB
  -> aspect-ratio resize and 768 x 768 letterbox
  -> normalize to Float32 [0, 1]
  -> run both models
  -> decode YOLO outputs
  -> confidence filtering
  -> class-aware NMS
  -> restore boxes to original-frame coordinates
  -> associate one playhead with the active progress bar
  -> temporal smoothing
  -> return detections
```

Both models use the same source frame and letterbox transformation.

## Preprocessing

The model was trained, validated, exported, and tested with the complete source
screenshot/frame supplied to the model. There is no separate runtime crop for
the browser window, player, control strip, or application UI. The complete
source image is resized with its aspect ratio preserved and letterboxed
directly to `768 x 768`. `scripts/infer.py` implements this exact path.

For source width `W` and height `H`:

```text
gain  = min(768 / W, 768 / H)
new_W = round(W * gain)
new_H = round(H * gain)
dw = 768 - new_W
dh = 768 - new_H
left = round(dw / 2 - 0.1)
top  = round(dh / 2 - 0.1)
```

Then:

1. Convert the frame to RGB.
2. Resize to `new_W x new_H` with bilinear interpolation.
3. Center-pad to `768 x 768` using RGB `(114, 114, 114)`.
4. Convert to Float32 and divide by `255.0`.
5. Save `gain`, `left`, and `top` for coordinate restoration.

Input tensors:

| Format | Shape | Layout | Dtype |
|---|---|---|---|
| ONNX | `[1, 3, 768, 768]` | RGB NCHW | Float32 |
| TFLite | `[1, 768, 768, 3]` | RGB NHWC | Float32 |

This letterbox procedure applies to every source resolution and orientation.

## Output decoding

Controls/progress output:

```text
[1, 17, 12096]
```

Playhead output:

```text
[1, 5, 12096]
```

The layout is:

```text
[batch, 4 xywh channels + class-score channels, candidates]
```

Transpose to `[12096, 4 + class_count]`. Each candidate contains:

```text
center_x, center_y, width, height, class scores...
```

The exports use class scores directly and do not contain a separate
objectness channel.

Postprocessing values:

```text
controls/progress confidence = 0.25
playhead confidence          = 0.20
class-aware NMS IoU          = 0.70
maximum detections           = 300
```

For each model:

1. Select the highest class score and class index for each candidate.
2. Apply the appropriate confidence threshold.
3. Convert `xywh` to `x1,y1,x2,y2`.
4. Apply class-aware NMS.

## Coordinate restoration

Restore every box after NMS:

```text
x_source = (x_letterbox - left) / gain
y_source = (y_letterbox - top) / gain
```

Apply the formula to both corners and clip the result to the source-frame
width and height.

## Combine the model outputs

Return ordinary controls and `progress_bar` from the controls/progress model.
Return `playhead` from the playhead model.

For a frame with one active player:

1. Select the progress bar with the largest `confidence * width`.
2. Create a temporary association region around it:

   ```text
   horizontal margin = max(4 * max(W,H)/768, 1% of bar width)
   vertical margin   = max(6 * max(W,H)/768, 2 * bar height)
   ```

3. Keep playhead candidates whose centers fall inside that region.
4. Select the highest-confidence candidate. Use distance to the bar's vertical
   center as the tie breaker.
5. Return at most one playhead for the selected bar.

The temporary association region does not modify the returned progress-bar
coordinates.

## Temporal smoothing

Temporal smoothing was not part of model inference or validation. It is a
recommended application-side step for live video. Apply it after detections
are restored to source-frame coordinates.

Recommended starting behavior:

- Match tracks only within the same class.
- Match ordinary controls with IoU `>= 0.30`.
- Confirm a new ordinary control when detected in 2 of the last 3 frames.
- Retain a confirmed ordinary control for up to 2 missed frames.
- Smooth ordinary controls and the progress bar with:

  ```text
  smoothed_box = 0.65 * current_box + 0.35 * previous_box
  ```

- Keep one playhead track for the selected progress bar.
- Smooth playhead x with:

  ```text
  smoothed_x = 0.85 * current_x + 0.15 * previous_x
  ```

- Retain a missing playhead for up to 1 frame.
- Replace mutually exclusive states when the new state appears:
  `play/pause`, `volume_on/volume_muted`, and
  `fullscreen/exit_fullscreen`.
- Reset tracks after orientation, source-resolution, active-player, or major
  control-surface changes.

Adjust confirmation and retention frame counts to the application's frame
rate while keeping the same behavior.

## Integration checklist

- [ ] Load a matching controls/progress and playhead model pair.
- [ ] Use RGB 768 letterbox preprocessing without distortion.
- [ ] Use NCHW for ONNX or NHWC for TFLite.
- [ ] Decode class scores without an objectness multiplier.
- [ ] Apply class-aware NMS.
- [ ] Restore detections to original-frame coordinates.
- [ ] Preserve the supplied class indices.
- [ ] Return controls/progress from the first model and playhead from the
      specialist.
- [ ] Return at most one playhead for the active progress bar.
- [ ] Apply and reset temporal tracks as described above.
- [ ] Verify portrait, landscape, rotation, regular video and Shorts.

## Reference files

```text
HANDOVER.md            implementation instructions
model_contracts.json   machine-readable model contract
tflite_contracts.json  TFLite tensor contract
manifest.json          model filenames and hashes
scripts/infer.py       complete reference inference path
youtube_tflite_visuals/ numbered-box TFLite examples with a side legend
```
