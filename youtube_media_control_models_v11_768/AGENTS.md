# Integration instructions for Codex

Before implementing the detector, read `HANDOVER.md` completely.

Use these files as the source of truth:

- `HANDOVER.md`
- `model_contracts.json`
- `tflite_contracts.json`
- `scripts/infer.py`

Implement the supplied two-model 768 pipeline exactly. Preserve preprocessing,
class indices, output decoding, class-aware NMS, coordinate restoration,
progress-bar/playhead association, and temporal smoothing.

The established model input is the complete supplied screenshot/frame followed
by direct 768 letterboxing. Do not add a separate browser, player, or UI-region
crop stage. Temporal smoothing is an application-side recommendation applied
after inference, not part of the model itself.

Complete the integration checklist in `HANDOVER.md` before delivery.
