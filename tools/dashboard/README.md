# MediaControl Collector Studio

From the project folder in PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\dashboard\server.py
```

Open <http://127.0.0.1:8000>. Keep the terminal open; Ctrl+C stops the server.
You can also double-click `Start Dashboard.cmd` in the project root. That launcher
starts the server in the background and opens the browser; its terminal can close.
Running the launcher again reuses a healthy existing server. Startup logs are
saved under `.test_tmp/dashboard/`. `/api/health` reports the server's process ID.
After a code update, restart the server and press Ctrl+F5 in the browser.

If recreating the environment, use Python 3.12, create `.venv`, then install the
tested dashboard dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools\dashboard\requirements.txt
```

## Capture, import, review

1. Open YouTube on the unlocked iPad and choose the correct format and capture
   state. Click **Capture One Frame**. When YouTube opens, show the controls.
   The native four-second preparation delay is followed by snapshot processing;
   the default XCTest timeout is 180 seconds, configurable with the environment
   variable `CAPTURE_TIMEOUT_SECONDS` before starting the dashboard.
2. A successful dashboard capture automatically imports only its one newly
   published record, with its screenshot and matching accessibility sidecar.
   Older unseen device records are excluded from that automatic import. If zero
   or multiple fresh IDs appear, automatic import stops for inspection.
   **Import New Captures** separately imports all unseen published device records.
   Previously imported or deleted record IDs are
   remembered in `data/.capture_import_history.json`; clearing collected folders
   does not make old captures return. Keep that history file.
3. Open **Inspect & Curate** on a frame. The gallery and editor hide exact and
   near-identical duplicate boxes. Diagnostics retains the original raw boxes.
   The editor automatically runs a recovery pass using that image's matching
   accessibility snapshot and original pixels. This works for existing captures
   as well as new ones; **Suggest Missing Labels** is not required to start it.
4. New native annotations marked `review_required` also open as pending
   suggestions, outside the accepted annotation list. Accept or reject each.
   The recovery pass corrects older pending box geometry and adds missing
   candidates. Click **Suggest Missing Labels** to also try reviewed examples.
   Use **Zoom** to check the pixels at 200%, then choose **Accept** or **Reject**.
   **Fit Image** returns to the whole screenshot.
5. Click **Apply Annotation Rules** to propose box changes for existing controls.
   Accepting a replacement preserves the original interaction region.
   **Show original tap regions** overlays those regions.
   Draw boxes for visible controls that are still missing. Use **Edit** to change
   an existing class, enter pixel coordinates, or drag a replacement rectangle.
6. Save as **needs_review** until you have checked the entire image. Choose
   **approved** or **resolved** after reviewing it. Suggestions left pending
   prevent saving an approved/resolved result. Edits reset the status to review.
   Saving as needs_review retains pending suggestions for the next review session.

Auto-capture waits for each complete capture and import, then pauses for the
selected number of seconds. It stops when a capture fails or needs attention.
The dashboard checks the final XCTest summary as well as process exit status;
an exit code of zero alone is not evidence of a successful capture.

The class picker uses the source annotation vocabulary independently of any
trained model. It does not imply those controls are visible or detected. Do not
add a playhead, pause button or skip-ad button when it is absent from the image.

The recovery pass localizes glyphs inside their accessibility regions. It checks
both diagonal fullscreen arrows instead of sizing the box from a single bright
patch. Autoplay uses the same `autoplay` class for both states. The full visible
toggle is boxed and the review list displays the observed on/off state.

For sliders, recovery requires a thin pixel stripe with visible boundaries
above and below, accepting red plus gray/white segments together. For an embedded
YouTube player, it can also trace the translucent track where the player meets the
page below. This uses the observed player viewport rather than the capture's
folder label. It derives a playhead only from a visible round thumb attached on
both sides of the stripe, including the part outside the accessibility hit
region. Playback percentage and timestamps never supply a thumb position.
Ambiguous tracks remain for review.

YouTube ad captures distinguish the player menu (`more_options`) from the
sponsored card or panel menu (`ad_more_options`). Ad actions use
`visit_advertiser`, `ad_cta`, `ad_info`, and `hide_content` alongside
`like` and `share` when those controls are visible. `ad_progress_bar`
frames the yellow ad track. The two fullscreen labels are resolved from the
arrows on the visible button. The recovery pass also checks the video action
row for the three-dot menu near Save and the Shorts toolbar for the back arrow.
In the Shorts toolbar, the visible Search icon anchors the speaker and
three-dot menu checks. The speaker's waves or mute X determine its state.
If the Shorts player shows its center playback overlay, the visible triangle
or paired bars determine whether it gets a `play` or `pause` box.
The Shorts `pause` box surrounds both visible bars with about 18% padding
per side, matching full-video Pause. The circle helps verify the overlay.
Raw iPad records and curated annotations are unchanged by automatic detection;
its candidates and diagnostics are saved under `curation/detections/`.

## Annotation storage

The existing annotation contract is preserved:

- `annotations.jsonl`: original source records, unchanged by review.
- `curation/annotation_overrides.json`: saved curated annotations. When present,
  `curated_annotations` replaces the raw annotation list, including an empty list.
- `curation/annotation_audit.jsonl`: exact before/after saved overrides and
  suggestion decisions.
- `curation/proposals/{record_id}.json`: predictions and matcher diagnostics;
  never automatically effective annotations or training ground truth.
- `accessibility/{record_id}.json`: complete native accessibility sidecar, when
  the native collector supplies one. Unmatched elements are retained.

`effective_annotations` in the API keeps the source handover contract.
`display_annotations` is a validated, deduplicated copy used by the dashboard.
Saving curation deduplicates that curated copy without rewriting raw data.
Exported handover ZIPs include snapshots and proposals in their separate folders.

## Visible target geometry

Accessibility labels can identify a control even when its frame covers the
icon, adjacent text and touch padding. Those labels remain useful metadata.
The visible annotation box uses a separate class-specific policy:

| Target | Policy |
| --- | --- |
| Like, comments, share, save, remix, search, playback and other icon classes | `icon_only`: complete visible glyph with about 18% padding per side, limited by nearby labels and controls |
| Subscribe / Subscribed | `visible_button`: full visible button, including the label and background |
| Skip Ad | `text_and_icon`: its text and adjacent skip glyph together; availability rules retained |
| Progress bar / playhead | `visible_track` keeps full track width and centers a box at least about 1.35% of the original image height on it. `visible_marker` covers the separate visible thumb with its center inside the track box. |
| Sound / channel | `artwork`: corresponding thumbnail/artwork |
| Autoplay | `visible_toggle`: the full capsule with `autoplay_state` set to `on` or `off` when verified from pixels |
| Time display / more videos | `text_readout` / `composite_control`: explicit class targets |

Apple documents accessibility frames as the element's screen rectangle, and
Android documents UI Automator bounds as object bounds. The dashboard therefore
keeps these separate from a visually localized glyph:
[Apple accessibilityFrame](https://developer.apple.com/documentation/uikit/uiaccessibilityelement/accessibilityframe),
[Android UiObject2](https://developer.android.com/reference/androidx/test/uiautomator/UiObject2).

`interaction_xyxy` holds the original tap region, `visible_content_xyxy` holds
the observed pixel bounds, and `visual_xyxy` holds the final annotation box.
The dashboard projects `visual_xyxy` into `xyxy` for review and curation.
Both are original-image pixels. Optional sidecar `visual_xyxy` also uses pixels
even when accessibility `frame` uses points; it must not be scaled twice.

Pixel localization proposes isolated foreground components within a known
control region. Play uses the visible triangle, while Previous and Next use
the triangle and its adjacent vertical bar. This checks dimmed glyphs against
their local background. Ambiguous shapes can be outlined manually.

Raw accessibility icon/Subscribe regions require geometry review before they
can be approved. Use a replacement proposal or check the target with **Edit**
and **Apply Box Changes**. Manual edits mark the geometry reviewed. Pixel/model
proposals require acceptance; they do not automatically become ground truth.

The picker includes the additional reference classes `minimize_player`, `cast`,
`hide_controls`, `subscribe`, `channel`, `download`, and `report_problem`. The
source class vocabulary is not limited to an old trained model's output classes.

`MediaCollector_iOS_Developer_Prompts_Updated.txt` contains the two updated native
developer prompts, including the user-confirmed Subscribe and Skip Ad policies.

**View capture diagnostics** shows the raw record, saved override, quality
report, complete available snapshot, and last generated suggestions.

## Detection and its limits

The previously supplied YouTube media-control model is legacy and disabled by
default. The current suggestion path uses complete accessibility data when
available, pixel localization of known control regions, and reviewed icon
examples. The pipeline does not load legacy model code or weights by default.
Its API reports `model_status` and an empty `model_classes` when disabled,
including if model sessions had previously been cached in the process.

The visual example matcher uses conservative multi-scale edge matching with
the supplied handover examples and approved/resolved local curations. It learns
additional appearances as you approve them. Artwork such as sound thumbnails
is unsuitable for reusable icon matching and is excluded from that fallback.
Subscribe and Skip Ad are deliberate text/control exceptions; their reviewed
templates are not converted into ordinary icon-only templates.

Different iPad icon styles can still be missed. In verification on the earlier
1668x2388 iPad screenshot, the reviewed-example matcher proposed the missing
more-options icon. This does not establish recall for the other visible
controls. Use manual review for the remaining boxes.
The geometry stage separately localized Search, Save and Remix in that capture,
including the translucent Remix glyph; these are reviewable geometry proposals,
not evidence of full automatic class recall on iPads.
No replacement neural detector is configured or validated. Full automatic
coverage needs representative reviewed iOS data and a separately validated
detector whose class map matches the source contract. The developer handoff
does not require the old model or its tensor/class-map contract.

For an explicit historical comparison only, the untouched legacy adapter can
be enabled with `MEDIACOLLECTOR_ENABLE_LEGACY_MODEL=1`. Its optional dependency
is in `requirements-legacy-model.txt`; normal dashboard setup does not need
ONNX Runtime. When opted in, it retains the supplied two-model full-frame
contract and supports YouTube captures only. This is not the recommended
capture pipeline or a validated replacement detector.

Captures are independent still images. Temporal smoothing is for consecutive
video frames and is not applied across these records. The live preview remains
a preview; this change does not add live video detection or tracking.

## Native iPad handover

The developer handover supplies `MediaCollector.xcodeproj` and the signed USB
kit. The [Capture Runner patch](../ios/CaptureRunnerHittability.patch) records
the source changes the developer tested on a landscape ad with its advertiser
panel open. It preserves the capture when XCTest cannot find a valid tap point
for an element. To apply it to the original handover, run `git apply` with the
patch path from the `MediaCollector/` folder that contains `Capture.swift`.
The updated source archive already contains the patch. The
`tools/ios/CaptureAccessibilitySnapshot.swift` example remains an integration
reference.

The native runner records enumerated elements, including unmatched/non-hittable
ones, and the complete available XCTest debug tree. Its handoff publishes the
image and sidecar before appending the record. The companion stores captures at:

```text
Documents/data/platform_specialist_yolo/{platform}/{device}/{format}/collected/
  images/{record_id}.png
  accessibility/{record_id}.json
```

The XCTest runner and collector have separate sandboxes, connected by the
handover's durable packet/companion bridge. A dashboard capture validates the
schema, record ID, exact PNG SHA256 and actual image dimensions across all three
artifacts before importing. Failed imports remain retryable.

The snapshot identifies the image with SHA256 and records its coordinate space.
The PC matcher uses exact normalized identifiers before supported action labels,
handles count suffixes, distinguishes like/dislike and mute/unmute state, and
converts screen points using the supplied full-screen rectangle. Icon and
Subscribe candidates need actual visual geometry, not just a matched label and
tap frame. Non-hittable candidates can be proposed only after pixel localization
or explicit visual geometry; otherwise they remain in diagnostics. `isHittable`
is a conservative proxy, not proof that an element is visually present.
Sequential accessibility enumeration can also
span an animation, so every proposal still needs image review.

The supplied evidence is useful development evidence, not complete annotation
coverage: the final portrait example misses Replay and the visible playhead;
the supplied landscape example has no raw annotations. Those visible targets
still need manual review or native improvements. Building and signing changes
to the supplied project requires a Mac and Xcode. This Windows integration does
not rebuild the signed apps or add a replacement neural detector.

A live Windows-to-iPad test on 2026-11-19 passed in 22.749 seconds and published
`youtube_ios_615F0A73-1985-407E-84EF-BC746D977C0B`. The dashboard imported exactly
that one Shorts capture, excluding seven older device records. Its 1668 by 2388
PNG, raw record and sidecar agree by SHA256 and dimensions; the sidecar contains
101 elements. Its nine native annotations are pending review, with no duplicate
or invalid boxes. A repeat scoped import transferred zero records. The Windows
suggestion pass can propose mute/search boxes and a channel-avatar candidate;
these remain unapproved suggestions. Sequential snapshot timing is explicitly
marked uncertain by the native capture.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest tools.test_annotation_geometry tools.test_annotation_quality tools.test_sync_from_device tools.test_legacy_model_config tools.test_capture_workflow tools.test_automatic_detection -v
node tools\test_dashboard_client.cjs
```

The checks use temporary collections, preserve raw records, verify review and
deduplication behavior, and cover repeated imports, deletion tombstones, transfer
retries, matching and mismatched accessibility snapshots, and template matching.
No deleted captures are restored to the active gallery by these checks.

Legacy model smoke checks performed earlier do not establish recognition
accuracy for the current workflow. Native iPad compilation, temporal video
tracking and any replacement neural detector remain outside the verified
still-image pipeline. Tests check that disabled legacy inference is never
called or advertised, including when old runtime sessions are cached.
