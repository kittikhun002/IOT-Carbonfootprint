"""Universal Meter Reader: YOLO (Localization) + TFLite CNN (Recognition & Rolling Transition).

Supports Electric (5-6 digits), Gas (7-8 digits), and Water (4-8 digits) meters.
Pipeline:
1. YOLO detects digit wheel positions (with 2-pass auto-zoom for high-res images).
2. Missing wheels are auto-extrapolated using grid step analysis.
3. Each wheel ROI is cropped and fed into the CNN to detect stable vs rolling digits.
4. Red dials are auto-detected for decimal placement (Gas/Water: 3 decimals, Elec: integer/1 decimal).
5. Mechanical Gear Transition logic resolves rolling digits into [before -> after] states.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

# ---------- Configuration ----------
YOLO_MODEL_PATH = "best.pt"
CNN_MODEL_PATH = "AI-on-the-edge-device__manual-setup__v16.1.0/sd-card/config/dig-class11_1910_s2_q.tflite"
FALLBACK_CNN_PATH = "dig-class100-0182-s2_q.tflite"
YOLO_CONFIDENCE = 0.12
YOLO_IOU = 0.3
CNN_CONFIDENCE_THRESHOLD = 0.70
SAVE_DEBUG_IMAGE = True
# -----------------------------------


class DigitCNN:
    """Wrapper for AI-on-the-edge TFLite Digit CNN models."""

    def __init__(self, model_path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.in_detail = self.interpreter.get_input_details()[0]
        self.out_detail = self.interpreter.get_output_details()[0]
        _, self.h, self.w, _ = self.in_detail["shape"]

    def predict(self, roi):
        if roi is None or roi.size == 0:
            return {"value": "N", "digit": None, "confidence": 0.0}

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (int(self.w), int(self.h)), interpolation=cv2.INTER_AREA).astype(np.float32)
        input_tensor = resized.astype(self.in_detail["dtype"])[np.newaxis, ...]

        self.interpreter.set_tensor(self.in_detail["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.out_detail["index"])[0].reshape(-1).astype(np.float32)

        # Dequantize
        scale, zero_pt = self.out_detail["quantization"]
        scores = (output - zero_pt) * scale if scale else output
        class_idx = int(np.argmax(scores))

        # Softmax confidence
        exp_s = np.exp(scores - np.max(scores))
        conf = float(exp_s[class_idx] / np.sum(exp_s))

        if class_idx == 10:  # Class 10 = Rolling transition (NaN)
            return {"value": "N", "digit": None, "confidence": conf}
        return {"value": str(class_idx), "digit": class_idx, "confidence": conf}


def is_red_roi(roi):
    """Detect if an ROI is a red decimal wheel."""
    if roi is None or roi.size == 0:
        return False
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 60, 50]), np.array([10, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([160, 60, 50]), np.array([180, 255, 255]))
    return ((cv2.countNonZero(m1) + cv2.countNonZero(m2)) / float(roi.shape[0] * roi.shape[1])) > 0.10


def get_yolo_detections(image_path_or_img, conf=YOLO_CONFIDENCE, iou=YOLO_IOU):
    """Run YOLO and return detected digit bounding boxes (classes 0-9)."""
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL_PATH)
    results = model(image_path_or_img, iou=iou, conf=conf, verbose=False)
    detections = []

    for r in results:
        for b in r.boxes:
            label = r.names[int(b.cls[0])]
            if label.isdigit():
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append({
                    "label": label, "conf": float(b.conf[0]),
                    "x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0,
                    "box": (x1, y1, x2, y2)
                })
    return sorted(detections, key=lambda d: d["x"])


def auto_zoom_highres(image, detections):
    """Auto-zoom onto digit area if high-resolution photo is detected."""
    if not detections:
        return image, detections
    h, w = image.shape[:2]
    if max(w, h) > 1500:
        x1s, y1s = [d["box"][0] for d in detections], [d["box"][1] for d in detections]
        x2s, y2s = [d["box"][2] for d in detections], [d["box"][3] for d in detections]
        min_x, max_x, min_y, max_y = min(x1s), max(x2s), min(y1s), max(y2s)
        span_w, span_h = max_x - min_x, max_y - min_y

        if span_w < w * 0.60:
            cx1, cy1 = max(0, min_x - int(span_w * 0.4)), max(0, min_y - int(span_h * 1.0))
            cx2, cy2 = min(w, max_x + int(span_w * 0.4)), min(h, max_y + int(span_h * 1.0))
            crop_dets = get_yolo_detections(image[cy1:cy2, cx1:cx2], conf=0.10)
            if len(crop_dets) >= len(detections):
                for d in crop_dets:
                    bx1, by1, bx2, by2 = d["box"]
                    d["box"] = (bx1 + cx1, by1 + cy1, bx2 + cx1, by2 + cy1)
                    d["x"], d["y"] = (d["box"][0] + d["box"][2]) / 2.0, (d["box"][1] + d["box"][3]) / 2.0
                return image, crop_dets
    return image, detections


def cluster_and_complete_wheels(detections, img_shape, expected_digits=None, meter_type="auto"):
    """Cluster YOLO detections into vertical wheel columns and fill any missing positions."""
    if not detections:
        return []

    # Dynamic clustering
    widths = [d["box"][2] - d["box"][0] for d in detections]
    x_thresh = max(10.0, float(np.median(widths)) * 0.60)
    groups = []
    for d in detections:
        if not groups or abs(d["x"] - np.mean([x["x"] for x in groups[-1]])) > x_thresh:
            groups.append([d])
        else:
            groups[-1].append(d)

    if len(groups) < 2:
        return groups

    # Extrapolate missing wheels using step distance
    h, w = img_shape[:2]
    centers = [float(np.mean([d["x"] for d in g])) for g in groups]
    dxs = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
    med_dx = float(np.median(dxs)) if dxs else 0
    avg_w = float(np.median([b[2] - b[0] for g in groups for b in [d["box"] for d in g]]))
    avg_h = float(np.median([b[3] - b[1] for g in groups for b in [d["box"] for d in g]]))
    avg_y = float(np.median([d["y"] for g in groups for d in g]))

    # 1. Fill gaps in between
    new_groups = []
    for i in range(len(groups)):
        new_groups.append(groups[i])
        if i < len(groups) - 1 and med_dx > 0:
            c1, c2 = float(np.mean([d["x"] for d in groups[i]])), float(np.mean([d["x"] for d in groups[i+1]]))
            gap = c2 - c1
            missing = int(round(gap / med_dx)) - 1
            for m in range(1, missing + 1):
                ix = c1 + (m * gap / (missing + 1))
                new_groups.append([{"label": "?", "conf": 0.0, "x": ix, "y": avg_y,
                                    "box": (max(0, int(ix - avg_w/2)), max(0, int(avg_y - avg_h/2)),
                                            min(w, int(ix + avg_w/2)), min(h, int(avg_y + avg_h/2)))}])
    groups = new_groups

    # 2. Fill boundary wheels if target is known
    target = expected_digits or (8 if meter_type == "gas" or (meter_type == "water" and len(groups) == 7) else None)
    if target and len(groups) < target and med_dx > 0:
        centers = [float(np.mean([d["x"] for d in g])) for g in groups]
        rx = centers[-1] + med_dx
        if rx + avg_w/2 < w:
            groups.append([{"label": "?", "conf": 0.0, "x": rx, "y": avg_y,
                            "box": (max(0, int(rx - avg_w/2)), max(0, int(avg_y - avg_h/2)),
                                    min(w, int(rx + avg_w/2)), min(h, int(avg_y + avg_h/2)))}])
    return groups


def make_wheel_rois(image, groups):
    """Build padded ROI images for each wheel position."""
    h, w = image.shape[:2]
    rois = []
    for i, g in enumerate(groups, 1):
        boxes = [d["box"] for d in g]
        x1, y1 = max(0, min(b[0] for b in boxes) - 4), max(0, min(b[1] for b in boxes) - 4)
        x2, y2 = min(w, max(b[2] for b in boxes) + 4), min(h, max(b[3] for b in boxes) + 4)
        crop = image[y1:y2, x1:x2]
        rois.append({"pos": i, "box": (x1, y1, x2, y2), "crop": crop, "group": g, "is_red": is_red_roi(crop)})
    return rois


def resolve_readings(rois, cnn_preds):
    """Ensemble YOLO + CNN predictions with Mechanical Transition Logic."""
    results = []
    has_transition = False

    for roi, cnn in zip(rois, cnn_preds):
        group = roi["group"]
        roi_h = roi["box"][3] - roi["box"][1]
        distinct = [d for d in group if d["conf"] > 0]
        distinct = sorted(distinct, key=lambda d: d["y"])
        
        has_two_yolo = len(distinct) >= 2 and distinct[0]["conf"] >= 0.25 and distinct[-1]["conf"] >= 0.25
        best_yolo = max(group, key=lambda d: d["conf"], default={"label": "?", "conf": 0.0})
        has_cnn_trans = (cnn.get("value") == "N" and cnn.get("confidence", 0) >= CNN_CONFIDENCE_THRESHOLD and best_yolo["conf"] < 0.65)

        if has_two_yolo or has_cnn_trans:
            has_transition = True
            bot_lbl = distinct[-1]["label"] if len(distinct) >= 2 else best_yolo["label"]
            enter_dig = int(bot_lbl) if bot_lbl.isdigit() else 0
            exit_dig = (enter_dig - 1) % 10
            results.append({
                "pos": roi["pos"], "is_transition": True, "display": f"[{exit_dig}->{enter_dig}]",
                "val_before": str(exit_dig), "val_after": str(enter_dig),
                "desc": f"กำลังหมุน [{exit_dig} -> {enter_dig}]", "is_red": roi["is_red"]
            })
        else:
            dig = best_yolo["label"] if best_yolo["conf"] >= 0.35 else (str(cnn.get("digit")) if cnn.get("digit") is not None else best_yolo["label"])
            results.append({
                "pos": roi["pos"], "is_transition": False, "display": dig,
                "val_before": dig, "val_after": dig,
                "desc": f"เลขนิ่ง [{dig}]", "is_red": roi["is_red"]
            })
    return results, has_transition


def format_output(results, meter_type="auto"):
    """Format final reading with correct decimals and units."""
    n = len(results)
    red_decimals = sum(1 for r in reversed(results) if r["is_red"])
    if red_decimals == 0:
        red_decimals = 3 if meter_type in ("gas", "water") or (meter_type == "auto" and n >= 7) else 0

    def add_dec(s, d):
        return f"{s[:-d]}.{s[-d:]}" if d > 0 and len(s) > d else s

    raw_b, raw_a = "".join(r["val_before"] for r in results), "".join(r["val_after"] for r in results)
    unit = " m³" if meter_type in ("gas", "water") or (meter_type == "auto" and n >= 7) else " kWh"
    return {"raw_b": raw_b, "raw_a": raw_a, "fmt_b": f"{add_dec(raw_b, red_decimals)}{unit}", "fmt_a": f"{add_dec(raw_a, red_decimals)}{unit}"}


def save_debug_image(img_path, img, rois, results):
    canvas = img.copy()
    for roi, res in zip(rois, results):
        x1, y1, x2, y2 = roi["box"]
        color = (0, 0, 255) if roi["is_red"] else ((0, 165, 255) if res["is_transition"] else (0, 255, 0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"#{res['pos']} {res['display']}", (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    out = Path("review") / f"cnn_{Path(img_path).name}"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), canvas)
    return out


def read_meter(image_path, expected_digits=None, meter_type="auto"):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"❌ ไม่พบไฟล์ภาพ: {image_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ เปิดภาพไม่ได้: {image_path}")
        return

    # 1. YOLO Localization & Auto-Zoom
    dets = get_yolo_detections(str(image_path))
    image, dets = auto_zoom_highres(image, dets)

    # 2. Wheel Clustering & Extrapolation
    groups = cluster_and_complete_wheels(dets, image.shape, expected_digits, meter_type)
    rois = make_wheel_rois(image, groups)

    print(f"\n📂 กำลังอ่านภาพ: {image_path.name} (ประเภทมิเตอร์: {meter_type})")
    print(f"--- ตรวจพบทั้งหมด {len(rois)} ช่อง ---")
    for r in rois:
        cand = ", ".join(f"{d['label']}({d['conf']:.2f})" for d in r["group"] if d["conf"] > 0) or "Auto Extrapolated"
        red = " [🔴 หลักทศนิยม]" if r["is_red"] else ""
        print(f"ช่อง {r['pos']}: {cand}{red}")

    # 3. CNN Recognition
    cnn_path = Path(CNN_MODEL_PATH) if Path(CNN_MODEL_PATH).exists() else Path(FALLBACK_CNN_PATH)
    cnn_preds = []
    if cnn_path.exists():
        cnn = DigitCNN(cnn_path)
        cnn_preds = [cnn.predict(r["crop"]) for r in rois]
    else:
        cnn_preds = [{}] * len(rois)

    # 4. Resolve & Format
    results, has_trans = resolve_readings(rois, cnn_preds)
    fmt = format_output(results, meter_type)

    print("\n--- ผลการอ่านค่ามิเตอร์ (YOLO + CNN + Mechanical Logic) ---")
    for r in results:
        print(f"ช่อง {r['pos']}: {r['desc']}")

    print(f"\n📌 ค่าที่อ่านได้ (ดิบ): {''.join(r['display'] for r in results)}")
    if has_trans:
        print(f"📌 ช่วงการเปลี่ยนค่า: {fmt['raw_b']} ➔ {fmt['raw_a']}")
        print(f"📌 ค่าตัวเลขพร้อมหน่วย: {fmt['fmt_b']} ➔ {fmt['fmt_a']}")
    else:
        print(f"📌 ค่าตัวเลขพร้อมหน่วย: {fmt['fmt_b']}")

    if SAVE_DEBUG_IMAGE:
        print(f"\n🖼️  บันทึกภาพ ROI สำหรับตรวจสอบ: {save_debug_image(image_path, image, rois, results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Meter Reader (Gas, Water, Electric)")
    parser.add_argument("image_path", help="Path to meter image")
    parser.add_argument("--digits", type=int, default=None, help="Expected number of digits (default: auto)")
    parser.add_argument("--type", choices=["auto", "elec", "gas", "water"], default="auto", help="Meter type")
    args = parser.parse_args()

    read_meter(args.image_path, expected_digits=args.digits, meter_type=args.type)
