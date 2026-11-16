# Icon Finder 🎯

A multi-paradigm UI icon detection and media control localization toolkit for mobile apps, web applications, and desktop platforms.

This repository provides multiple specialized approaches for locating, parsing, and classifying UI icons and media player controls:

| Approach | Technology | Best For | Notebook / Path |
|:---|:---|:---|:---|
| **YouTube Media Controls (V11 — 768px)** | Dual-model YOLO (Controls + Playhead Specialist) in TFLite / ONNX | Production YouTube player control detection across Desktop, Mobile, Tablet, and Shorts | [`YouTube_Media_Controls_Colab.ipynb`](./YouTube_Media_Controls_Colab.ipynb) / [`youtube_media_control_models_v11_768/`](./youtube_media_control_models_v11_768/) |
| **Qwen3-VL Vision-Language Model** | Qwen3-VL-2B-Instruct (4-bit NF4) | Zero-shot visual reasoning and semantic icon searching ("settings gear", "share icon") | [`Qwen3_VL_Icon_Finder_Colab.ipynb`](./Qwen3_VL_Icon_Finder_Colab.ipynb) |
| **OmniParser v2** | YOLOv8 Icon Detector + Florence-2 Captioner + EasyOCR | Full-screen UI layout parsing and candidate ranking | [`OmniParser_Colab_Fixedv2.ipynb`](./OmniParser_Colab_Fixedv2.ipynb) |
| **Universal Classical Icon Finder** | OpenCV Template & Contour Feature Matching | Fast, zero-GPU offline detection without neural network dependencies | [`universal_classical_icon_finder/`](./universal_classical_icon_finder/) |

---

## 🚀 YouTube Media Controls Model (V11 — 768px)

A decoupled two-model system optimized for video player interfaces:
1. **`youtube_controls_progress`** (13 classes): Detects `play`, `pause`, `volume_on`, `volume_muted`, `fullscreen`, `exit_fullscreen`, `previous`, `next`, `like`, `dislike`, `seek_backward`, `seek_forward`, and the full `progress_bar`.
2. **`youtube_playhead`** (1 class specialist): Dedicated detector solely for the `playhead` scrubber thumb, resolving occlusion and small-object suppression on top of seek tracks.

### Capabilities in `YouTube_Media_Controls_Colab.ipynb`:
- **Track 1 · Pretrained Inference**: Run the delivered Float32/Float16 TFLite or ONNX models on screenshots with 768px letterboxing, class-aware NMS, coordinate un-letterboxing, and playhead spatial association.
- **Track 2 · Retraining with YOLO11**: Bootstraps training datasets via pseudo-labeling, enforces UI-safe training rules (no flips/rotations), and exports back to ONNX and TFLite.

---

## 🧠 Qwen3-VL Zero-Shot Icon Finder

Located in [`Qwen3_VL_Icon_Finder_Colab.ipynb`](./Qwen3_VL_Icon_Finder_Colab.ipynb):
- Uses `Qwen/Qwen3-VL-2B-Instruct` in 4-bit NF4 quantization on Google Colab T4 GPUs.
- Accepts natural language queries (e.g. *"main play button in active video player; ignore thumbnails"*) and outputs pixel-accurate bounding boxes and click centers.

---

## ⚡ Classical Icon Finder

Located in [`universal_classical_icon_finder/`](./universal_classical_icon_finder/):
- Lightweight Python + OpenCV application with Streamlit UI and CLI interface.
- Includes platform profiles (`youtube.json`, `netflix.json`, `general.json`) and a reference icon library for deterministic matching.

---

## 📦 Repository Structure

```text
Icon-Finder/
├── YouTube_Media_Controls_Colab.ipynb     # YouTube dual-model inference & YOLO11 training
├── Qwen3_VL_Icon_Finder_Colab.ipynb       # Qwen3-VL 2B zero-shot semantic icon finder
├── OmniParser_Colab_Fixedv2.ipynb         # OmniParser v2 UI parser walkthrough
├── OmniParser_Colab_Fixed.ipynb           # OmniParser fixed pipeline
├── OmniParser_Colab_Walkthrough.ipynb     # OmniParser walkthrough
├── youtube_media_control_models_v11_768/  # V11 768px deployment package
│   ├── models/
│   │   ├── onnx/                          # Fixed 768x768 ONNX models
│   │   └── tflite/                        # Float32 and Float16 TFLite models
│   ├── scripts/infer.py                   # Reference Python inference implementation
│   ├── youtube_tflite_visuals/            # Reference test visuals and contact sheets
│   ├── HANDOVER.md                        # Formal engineering & tensor contracts
│   └── README.md                          # Model architecture specifications
└── universal_classical_icon_finder/       # Classical OpenCV-based detector & Streamlit app
```
