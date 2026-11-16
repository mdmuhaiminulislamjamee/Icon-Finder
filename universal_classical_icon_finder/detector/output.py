import csv
from io import BytesIO, StringIO
import json
from PIL import Image, ImageDraw, ImageFont


def results_json(results):
    return json.dumps(results, indent=2, ensure_ascii=False)


def flat_rows(results):
    return [{"icon": r["icon"], "platform": r["platform"], "confidence": r["confidence"],
             "method": r["method"], "raw_score": r["raw_score"], **r["bounding_box"],
             "center_x": r["center"]["x"], "center_y": r["center"]["y"], "scale": r["scale"],
             "template": r["template"], "processing_time_ms": r["processing_time_ms"],
             "image_width": r["image_size"]["width"], "image_height": r["image_size"]["height"],
             "supporting_methods": ";".join(r["supporting_methods"])} for r in results]


def results_csv(results):
    output = StringIO(newline="")
    rows = flat_rows(results)
    fields = list(rows[0]) if rows else ["icon", "platform", "confidence", "method", "raw_score", "x1", "y1", "x2", "y2",
                                        "center_x", "center_y", "scale", "template", "processing_time_ms",
                                        "image_width", "image_height", "supporting_methods"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def annotate(rgb, results):
    image = Image.fromarray(rgb).copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    colors = ["#18a579", "#6d5ce7", "#dc5d28", "#1685d2", "#bd3b83"]
    names = sorted({r["icon"] for r in results})
    for result in results:
        color = colors[names.index(result["icon"]) % len(colors)]
        x1, y1, x2, y2 = result["bounding_box"].values()
        draw.rectangle((x1, y1, x2-1, y2-1), outline=color, width=2)
        cx, cy = result["center"].values()
        draw.line((cx-3, cy, cx+3, cy), fill=color, width=1)
        draw.line((cx, cy-3, cx, cy+3), fill=color, width=1)
        label = f"{result['icon']} {result['confidence']:.2f}"
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2]-box[0]+8, box[3]-box[1]+8
        lx, ly = max(0, min(x1, image.width-tw)), max(0, y1-th)
        draw.rectangle((lx, ly, min(image.width, lx+tw), ly+th), fill=color)
        draw.text((lx+4, ly+2), label, fill="white", font=font)
    return image


def png_bytes(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()
