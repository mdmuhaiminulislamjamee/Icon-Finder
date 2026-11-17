#!/usr/bin/env python3
"""Reference inference for the delivered ONNX and TFLite model pairs.

The implementation intentionally uses only standard YOLO decoding. It never
creates or recovers a progress bar that was not emitted by the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


INPUT_SIZE = 768
PAD_VALUE = 114
CONTROL_NAMES = (
    "play",
    "pause",
    "volume_on",
    "volume_muted",
    "fullscreen",
    "exit_fullscreen",
    "previous",
    "next",
    "like",
    "dislike",
    "seek_backward",
    "seek_forward",
    "progress_bar",
)
PLAYHEAD_NAMES = ("playhead",)
COLORS = {
    name: tuple(int(value) for value in color)
    for name, color in zip(
        CONTROL_NAMES + PLAYHEAD_NAMES,
        (
            (70, 220, 70), (40, 190, 250), (250, 170, 40), (230, 80, 80),
            (200, 80, 240), (160, 70, 240), (230, 180, 60), (60, 180, 230),
            (80, 220, 170), (70, 140, 240), (190, 210, 50), (240, 120, 190),
            (255, 80, 200), (40, 60, 255),
        ),
    )
}


@dataclass
class Detection:
    box: np.ndarray
    score: float
    class_id: int
    label: str

    def as_dict(self) -> dict:
        return {
            "class_id": self.class_id,
            "class": self.label,
            "confidence": round(self.score, 6),
            "xyxy": [round(float(value), 3) for value in self.box],
        }


def letterbox(rgb: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Resize without distortion, then center-pad to 768 x 768."""
    height, width = rgb.shape[:2]
    gain = min(INPUT_SIZE / width, INPUT_SIZE / height)
    resized_width = round(width * gain)
    resized_height = round(height * gain)
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    dw = INPUT_SIZE - resized_width
    dh = INPUT_SIZE - resized_height
    left = round(dw / 2 - 0.1)
    right = round(dw / 2 + 0.1)
    top = round(dh / 2 - 0.1)
    bottom = round(dh / 2 + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(PAD_VALUE, PAD_VALUE, PAD_VALUE),
    )
    assert padded.shape[:2] == (INPUT_SIZE, INPUT_SIZE)
    return padded, gain, left, top


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return result


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    x1 = np.maximum(one[0], many[:, 0])
    y1 = np.maximum(one[1], many[:, 1])
    x2 = np.minimum(one[2], many[:, 2])
    y2 = np.minimum(one[3], many[:, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_one = max(0.0, one[2] - one[0]) * max(0.0, one[3] - one[1])
    area_many = np.maximum(0.0, many[:, 2] - many[:, 0]) * np.maximum(0.0, many[:, 3] - many[:, 1])
    return intersection / np.maximum(area_one + area_many - intersection, 1e-7)


def class_aware_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
    max_detections: int = 300,
) -> list[int]:
    kept: list[int] = []
    for class_id in np.unique(class_ids):
        candidates = np.flatnonzero(class_ids == class_id)
        candidates = candidates[np.argsort(scores[candidates])[::-1]]
        while len(candidates):
            selected = int(candidates[0])
            kept.append(selected)
            if len(candidates) == 1:
                break
            remaining = candidates[1:]
            candidates = remaining[box_iou(boxes[selected], boxes[remaining]) <= iou_threshold]
    kept.sort(key=lambda index: float(scores[index]), reverse=True)
    return kept[:max_detections]


class Runtime:
    def __init__(self, path: Path):
        self.path = path
        if path.suffix == ".onnx":
            import onnxruntime as ort

            self.kind = "onnx"
            self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self.input = self.session.get_inputs()[0]
            self.input_shape = list(self.input.shape)
        elif path.suffix == ".tflite":
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

            self.kind = "tflite"
            self.session = Interpreter(model_path=str(path), num_threads=4)
            self.session.allocate_tensors()
            self.input = self.session.get_input_details()[0]
            self.output = self.session.get_output_details()[0]
            self.input_shape = self.input["shape"].tolist()
        else:
            raise ValueError(f"Unsupported model format: {path}")

    def invoke(self, rgb_768: np.ndarray) -> np.ndarray:
        tensor = rgb_768.astype(np.float32) / 255.0
        if self.kind == "onnx":
            tensor = np.transpose(tensor, (2, 0, 1))[None]
            return self.session.run(None, {self.input.name: tensor})[0]
        tensor = tensor[None]
        self.session.set_tensor(self.input["index"], tensor)
        self.session.invoke()
        return self.session.get_tensor(self.output["index"])


def decode(
    raw: np.ndarray,
    names: tuple[str, ...],
    confidence: float,
    iou: float,
    gain: float,
    pad_left: int,
    pad_top: int,
    image_shape: tuple[int, int],
) -> list[Detection]:
    prediction = np.asarray(raw)[0]
    if prediction.shape[0] == 4 + len(names):
        prediction = prediction.T
    if prediction.shape[1] != 4 + len(names):
        raise ValueError(f"Unexpected output shape {raw.shape} for {len(names)} classes")

    class_scores = prediction[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(prediction)), class_ids]
    mask = scores >= confidence
    scores = scores[mask]
    class_ids = class_ids[mask]
    boxes = xywh_to_xyxy(prediction[mask, :4])
    if not len(boxes):
        return []
    keep = class_aware_nms(boxes, scores, class_ids, iou)
    boxes = boxes[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / gain
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / gain
    height, width = image_shape
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return [
        Detection(box, float(score), int(class_id), names[int(class_id)])
        for box, score, class_id in zip(boxes, scores, class_ids)
    ]


def select_one_playhead(
    controls: list[Detection], playheads: list[Detection], image_shape: tuple[int, int]
) -> list[Detection]:
    """Associate one real playhead detection with one real progress-bar detection."""
    bars = [item for item in controls if item.label == "progress_bar"]
    if not bars or not playheads:
        return []
    bar = max(bars, key=lambda item: item.score * max(1.0, item.box[2] - item.box[0]))
    x1, y1, x2, y2 = bar.box
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    coordinate_scale = max(image_shape) / INPUT_SIZE
    x_margin = max(4.0 * coordinate_scale, 0.01 * width)
    y_margin = max(6.0 * coordinate_scale, 2.0 * height)
    candidates = []
    for item in playheads:
        center_x = (item.box[0] + item.box[2]) / 2
        center_y = (item.box[1] + item.box[3]) / 2
        if x1 - x_margin <= center_x <= x2 + x_margin and y1 - y_margin <= center_y <= y2 + y_margin:
            candidates.append(item)
    if not candidates:
        return []
    center_y = (y1 + y2) / 2
    return [max(candidates, key=lambda item: (item.score, -abs((item.box[1] + item.box[3]) / 2 - center_y)))]


def get_video_type_flag(image_shape: tuple[int, int], detections: list[Detection]) -> int:
    """
    Determines whether the screenshot is a Regular / Long Video or YouTube Short.
    Returns:
        0: Regular / Long Video
        1: YouTube Short
    """
    height, width = image_shape[:2]
    labels = {item.label for item in detections}

    # If any regular controls are present -> 100% Long Video (0)
    regular_controls = {"fullscreen", "exit_fullscreen", "next", "previous", "seek_backward", "seek_forward"}
    if labels.intersection(regular_controls):
        return 0

    # If vertical action rail (like & dislike stacked on right edge) -> Short (1)
    likes = [item for item in detections if item.label == "like"]
    dislikes = [item for item in detections if item.label == "dislike"]
    if likes and dislikes:
        if (likes[0].box[0] > 0.70 * width) and (dislikes[0].box[0] > 0.70 * width):
            if dislikes[0].box[1] > likes[0].box[1]:
                return 1

    # Aspect ratio check: Phone vertical 9:16 (W/H < 0.75) -> Short (1)
    if (width / height) < 0.75:
        return 1

    # Default to Long Video (0)
    return 0


def _font(size: int, bold: bool = False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def render(
    bgr: np.ndarray,
    detections: list[Detection],
    title: str = "TFLite detections",
) -> np.ndarray:
    """Render the V1 handover format: numbered boxes plus a side legend."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(rgb)
    draw = ImageDraw.Draw(source)
    width, height = source.size
    line_width = max(2, round(width / 650))
    number_size = max(17, min(30, round(min(width, height) / 36)))
    number_font = _font(number_size, bold=True)

    for number, item in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(round(float(value))) for value in item.box]
        color = COLORS[item.label]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        text = str(number)
        bounds = draw.textbbox((0, 0), text, font=number_font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        pad = 5
        chip_x = max(0, min(width - text_width - pad * 2, x1))
        chip_y = max(0, y1 - text_height - pad * 2)
        draw.rounded_rectangle(
            (chip_x, chip_y, chip_x + text_width + pad * 2, chip_y + text_height + pad * 2),
            radius=4,
            fill=color,
        )
        draw.text(
            (chip_x + pad, chip_y + pad - bounds[1]),
            text,
            font=number_font,
            fill="white",
        )

    panel_width = max(390, min(570, round(width * 0.38)))
    output = Image.new("RGB", (width + panel_width, height), "#15181d")
    output.paste(source, (0, 0))
    panel = ImageDraw.Draw(output)
    margin = 22
    title_size = max(22, min(34, round(height / 34)))
    title_font = _font(title_size, bold=True)
    subtitle_font = _font(max(15, round(title_size * 0.58)))
    panel.text((width + margin, margin), title, font=title_font, fill="white")
    panel.text(
        (width + margin, margin + title_size + 8),
        "Controls + playhead",
        font=subtitle_font,
        fill="#b7c0cc",
    )
    panel.text(
        (width + margin, margin + title_size + 33),
        f"{len(detections)} detections",
        font=subtitle_font,
        fill="#b7c0cc",
    )

    list_top = margin + title_size + 72
    available = height - list_top - margin
    line_height = max(28, available // max(1, len(detections)))
    legend_font = _font(max(15, min(25, round(line_height * 0.55))), bold=True)
    for number, item in enumerate(detections, 1):
        y = list_top + (number - 1) * line_height
        panel.text(
            (width + margin, y),
            f"{number:>2}  {item.label}  {item.score:.2f}",
            font=legend_font,
            fill=COLORS[item.label],
        )
    return cv2.cvtColor(np.asarray(output), cv2.COLOR_RGB2BGR)


def model_paths(delivery: Path, backend: str, precision: str) -> tuple[Path, Path]:
    if backend == "onnx":
        return (
            delivery / "models/onnx/youtube_controls_progress.onnx",
            delivery / "models/onnx/youtube_playhead.onnx",
        )
    return (
        delivery / f"models/tflite/youtube_controls_progress_{precision}.tflite",
        delivery / f"models/tflite/youtube_playhead_{precision}.tflite",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--backend", choices=("tflite", "onnx"), default="tflite")
    parser.add_argument("--precision", choices=("float32", "float16"), default="float32")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--playhead-confidence", type=float, default=0.20)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    padded, gain, left, top = letterbox(rgb)
    delivery = Path(__file__).resolve().parents[1]
    controls_path, playhead_path = model_paths(delivery, args.backend, args.precision)
    controls_runtime = Runtime(controls_path)
    playhead_runtime = Runtime(playhead_path)
    controls = decode(
        controls_runtime.invoke(padded), CONTROL_NAMES, args.confidence, args.iou,
        gain, left, top, rgb.shape[:2],
    )
    playhead_candidates = decode(
        playhead_runtime.invoke(padded), PLAYHEAD_NAMES, args.playhead_confidence, args.iou,
        gain, left, top, rgb.shape[:2],
    )
    playhead = select_one_playhead(controls, playhead_candidates, rgb.shape[:2])
    detections = controls + playhead
    detections.sort(key=lambda item: (item.box[1], item.box[0]))

    is_short_flag = get_video_type_flag(rgb.shape, detections)

    payload = {
        "image": str(args.image),
        "image_size": [bgr.shape[1], bgr.shape[0]],
        "is_short": is_short_flag,
        "video_type": "Short" if is_short_flag == 1 else "Long",
        "backend": args.backend,
        "precision": args.precision if args.backend == "tflite" else "float32",
        "input_size": [INPUT_SIZE, INPUT_SIZE],
        "detections": [item.as_dict() for item in detections],
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        title = "TFLite detections" if args.backend == "tflite" else "ONNX detections"
        if not cv2.imwrite(str(args.save), render(bgr, detections, title=title)):
            raise RuntimeError(f"Could not save {args.save}")


if __name__ == "__main__":
    main()
