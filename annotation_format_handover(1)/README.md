# MediaControlDetector Annotation-Format Handover

This package demonstrates the current annotation format using three real,
curated examples from the MediaControlDetector dataset. It is intended to give
another engineer or Codex session enough context to understand the storage
contract without reading the complete dataset.

## Start here

Extract the ZIP and read this file, then inspect `CURATED_EXAMPLES.json`
alongside the three files under `images/`.

`CURATED_EXAMPLES.json` contains, for each selected image:

- record context;
- a summary of the original raw annotations;
- curation status and notes;
- the exact effective curated annotations loaded from the source dataset.

Match an entry's `packaged_image` field to the corresponding image. Annotation
coordinates use the original image's pixel dimensions.

## Annotation contract

The canonical dataset structure is:

```text
data/platform_specialist_yolo/{platform}/{device}/{content_format}/collected/
├── images/                         # canonical raw images
├── annotations.jsonl               # one raw source record per image
└── curation/
    ├── annotation_overrides.json   # effective curated corrections
    └── annotation_audit.jsonl      # append-only correction history
```

Important rules:

1. `annotations.jsonl` is the raw/source annotation record.
2. Incorrect raw annotations are corrected through curation; `raw` does not
   mean an incorrect label must remain effective.
3. If a record has `curated_annotations` in `annotation_overrides.json`, those
   are its effective annotations.
4. Otherwise, its effective annotations are `raw_annotations`.
5. Training labels are derived at runtime through a class-map configuration.
   The training class set is not another source annotation copy.
6. Coordinates use pixel-space `xyxy`: `[left, top, right, bottom]`.
7. Source annotations may preserve interaction, visual, and canonical boxes,
   descriptors, resource IDs, geometry policies, and visual evidence.

## Selected examples

### Desktop full video

`media_web_states_v2__youtube__sports__c3d12318af24__fhd_fullscreen_playing`

Shows a resolved fullscreen desktop player with a broad control vocabulary,
including transport, seek, volume, fullscreen-state, settings, and social UI.

### Phone short video

`youtube_android_000257`

Shows a portrait YouTube Short paused at a visible seek position with the muted
toolbar state. It includes the progress bar, playhead, action rail, sound,
search, more-options, and `volume_muted` annotations.

### Tablet short video

`youtube_android_000281`

Shows the same annotation schema applied to the tablet layout. This demonstrates
that platform/device geometry can differ while annotation naming and record
structure remain consistent.

## Package contents

```text
README.md                           this context document
CURATED_EXAMPLES.json               selected metadata and exact curated boxes
SHA256SUMS.txt                      integrity checksums
images/                             three explicitly authorized example copies
configs/annotation_examples.json   visible example-selection configuration
```

The image copies in this ZIP are handover examples only. The canonical images
inside the repository remain the source of truth.

`CURATED_EXAMPLES.json` is a handover snapshot, not a new source of truth.
Future corrections must still be made in the canonical collection's curation
files. The package should then be regenerated from those records.

## Integrity check

After extracting the package:

```bash
sha256sum -c SHA256SUMS.txt
```
