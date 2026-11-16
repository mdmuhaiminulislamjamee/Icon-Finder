# Validation

Validated on November 4, 2026 with Windows and Python 3.12.10. Exact package versions are listed in `requirements-tested.txt`.

- All 60 automated tests passed in the final run (12.33 seconds).
- Generic, YouTube, Netflix, and custom_demo profiles each detected an independently positioned play reference using the same core.
- New unknown profiles, shared identities, clone isolation, and exported/imported profiles were exercised in temporary directories.
- Desktop sample: all 8 placed controls recovered, including two play buttons at 1× and 1.5×. Mobile dark sample: all 6 placed controls recovered. Tests compare every returned box with synthetic ground truth.
- Original dimensions and box coordinates were checked, including image borders and a 1920 × 1080 screenshot. Duplicate detections were fused without removing spatially separate instances.
- The Streamlit application started, selected its desktop sample, ran detection, displayed eight matches, and invalidated stale results after a threshold change under Streamlit AppTest.
- Python compilation and dependency consistency checks passed. A local Streamlit server started on `127.0.0.1:8501`.

The available browser-control host could not connect, so a live browser screenshot was unavailable. UI behavior was verified using Streamlit's application test harness. The generated library PNG was visually inspected.

This verifies implementation behavior on controlled synthetic inputs. It does not establish recognition accuracy on real platform screenshots, pixel-perfect correspondence to the chat preview, or cross-OS execution. No real-platform accuracy percentage is claimed.
