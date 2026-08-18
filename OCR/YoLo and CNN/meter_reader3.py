"""Meter reader: YOLO finds digit positions; TFLite CNN + Mechanical Transition Logic reads digits.

Features:
1. YOLO locates the 5 wheel positions and candidate bounding boxes.
2. AI-on-the-edge-device CNN (dig-class11) classifies stable digits (0-9).
3. If CNN detects 'N' (NaN / rolling transition) or multiple vertical boxes:
   - Mechanical Transition Logic uses gear law (d_exit = (d_enter - 1) % 10)
     to accurately decode rolling digits (e.g. 3->4 and 9->0 on m14.jpg).
4. Handles steady digits seamlessly without false transitions (e.g. m6.JPG -> 04636).

Usage:
    python meter_reader3.py m14.jpg
    python meter_reader3.py m6.JPG
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ---------- Settings ----------
YOLO_MODEL_PATH = "best.pt"
CNN_MODEL_PATH = (
    "AI-on-the-edge-device__manual-setup__v16.1.0/sd-card/config/"
    "dig-class11_1910_s2_q.tflite"
)
EXPECTED_DIGITS = 5
YOLO_CONFIDENCE = 0.08
YOLO_IOU = 0.3
CNN_CONFIDENCE_THRESHOLD = 0.70
SAVE_DEBUG_IMAGE = True
CNN_PREPROCESSING = "rgb_0_255"
# ------------------------------


def load_interpreter(model_path):
    """Load a TFLite interpreter."""
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "ไม่พบ TFLite interpreter. ติดตั้ง TensorFlow ใน Python environment "
                "ที่ใช้รันไฟล์นี้ก่อน: pip install tensorflow"
            ) from exc

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    return interpreter


class DigitCNN:
    """Wrapper for AI-on-the-edge-device digit CNNs (dig-class11)."""

    def __init__(self, model_path):
        self.interpreter = load_interpreter(model_path)
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        _, self.height, self.width, self.channels = self.input_detail["shape"]

        if self.channels != 3:
            raise ValueError(
                f"CNN model ต้องรับภาพ RGB 3 channels แต่โมเดลนี้รับ {self.channels} channels"
            )
        self.class_count = int(np.prod(self.output_detail["shape"]))
        self.model_type = "dig-class11" if self.class_count == 11 else f"dig-class{self.class_count}"

    @staticmethod
    def _dequantize(tensor, detail):
        scale, zero_point = detail["quantization"]
        if scale:
            return (tensor.astype(np.float32) - zero_point) * scale
        return tensor.astype(np.float32)

    @staticmethod
    def _confidence(scores, index):
        """Convert the CNN's raw logits into a probability with softmax."""
        scores = scores.astype(np.float32)
        shifted = scores - scores.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        return float(probabilities[index])

    def _prepare_pixels(self, roi, preprocessing):
        if roi is None or roi.size == 0:
            raise ValueError("ROI ของตัวเลขว่าง")

        if preprocessing.startswith("bgr"):
            pixels = roi
        else:
            pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        pixels = cv2.resize(pixels, (int(self.width), int(self.height)), interpolation=cv2.INTER_AREA)
        pixels = pixels.astype(np.float32)
        if preprocessing.endswith("0_1"):
            pixels /= 255.0
        elif preprocessing.endswith("minus1_1"):
            pixels = pixels / 127.5 - 1.0
        return pixels

    def predict(self, roi, preprocessing="rgb_0_255"):
        pixels = self._prepare_pixels(roi, preprocessing)
        input_tensor = pixels.astype(self.input_detail["dtype"])[np.newaxis, ...]

        self.interpreter.set_tensor(self.input_detail["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])[0].reshape(-1)
        scores = self._dequantize(output, self.output_detail)
        class_index = int(np.argmax(scores))

        prediction = {"confidence": self._confidence(scores, class_index)}
        if class_index == 10:
            # dig-class11 uses class 10 as NaN / Rolling Transition
            prediction.update({"value": "N", "digit": None})
        else:
            prediction.update({"value": str(class_index), "digit": class_index})
        return prediction


def get_yolo_detections(image_path):
    """Return all digit boxes from YOLO."""
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL_PATH)
    results = model(image_path, iou=YOLO_IOU, agnostic_nms=False, conf=YOLO_CONFIDENCE, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls[0])]
            if label == "Reading Digit":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(
                {
                    "yolo_label": label,
                    "yolo_conf": float(box.conf[0]),
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "box": (x1, y1, x2, y2),
                }
            )

    return sorted(detections, key=lambda item: item["x"])


def cluster_wheel_positions(detections):
    """Cluster overlapping YOLO boxes into physical wheel positions by x-coordinate."""
    if not detections:
        return []

    widths = [item["box"][2] - item["box"][0] for item in detections]
    x_threshold = max(12.0, float(np.median(widths)) * 0.70)
    groups = []

    for detection in detections:
        if not groups:
            groups.append([detection])
            continue

        previous_center = float(np.mean([item["x"] for item in groups[-1]]))
        if abs(detection["x"] - previous_center) <= x_threshold:
            groups[-1].append(detection)
        else:
            groups.append([detection])

    return groups


def auto_complete_grid(groups, image_shape, expected_digits=5):
    """If one wheel is missing (e.g. 4 found of 5), extrapolate its position from grid spacing."""
    if len(groups) == expected_digits - 1 and len(groups) >= 3:
        centers = [np.mean([d["x"] for d in g]) for g in groups]
        dxs = [centers[i+1] - centers[i] for i in range(len(centers)-1)]
        median_dx = np.median(dxs)
        avg_w = np.median([b[2]-b[0] for g in groups for b in [d["box"] for d in g]])
        avg_h = np.median([b[3]-b[1] for g in groups for b in [d["box"] for d in g]])
        avg_y = np.median([d["y"] for g in groups for d in g])

        img_h, img_w = image_shape[:2]
        # Check if missing wheel is on the right
        right_x = centers[-1] + median_dx
        if right_x + avg_w/2 < img_w:
            nx1 = max(0, int(right_x - avg_w/2))
            ny1 = max(0, int(avg_y - avg_h/2))
            nx2 = min(img_w, int(right_x + avg_w/2))
            ny2 = min(img_h, int(avg_y + avg_h/2))
            groups.append([{
                "yolo_label": "?",
                "yolo_conf": 0.0,
                "x": right_x,
                "y": avg_y,
                "box": (nx1, ny1, nx2, ny2)
            }])
        # Check if missing wheel is on the left
        elif centers[0] - median_dx - avg_w/2 > 0:
            left_x = centers[0] - median_dx
            nx1 = max(0, int(left_x - avg_w/2))
            ny1 = max(0, int(avg_y - avg_h/2))
            nx2 = min(img_w, int(left_x + avg_w/2))
            ny2 = min(img_h, int(avg_y + avg_h/2))
            groups.insert(0, [{
                "yolo_label": "?",
                "yolo_conf": 0.0,
                "x": left_x,
                "y": avg_y,
                "box": (nx1, ny1, nx2, ny2)
            }])
    return groups


def make_wheel_rois(image, groups):
    """Build one tight ROI for every wheel position found by YOLO."""
    all_boxes = [item["box"] for group in groups for item in group]
    if not all_boxes:
        return []

    image_height, image_width = image.shape[:2]
    rois = []
    for index, group in enumerate(groups, start=1):
        boxes = [item["box"] for item in group]
        x1 = min(box[0] for box in boxes)
        y1 = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        padding_x = max(2, int((x2 - x1) * 0.08))
        padding_y = max(2, int((y2 - y1) * 0.05))
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image_width, x2 + padding_x)
        y2 = min(image_height, y2 + padding_y)
        rois.append(
            {
                "position": index,
                "box": (x1, y1, x2, y2),
                "yolo_candidates": group,
                "image": image[y1:y2, x1:x2],
            }
        )
    return rois


def resolve_wheel_readings(rois, cnn_predictions, groups):
    """Decode stable digits and rolling transitions using Mechanical Transition Logic."""
    wheel_results = []
    has_transition = False

    for idx, (roi, cnn_pred, group) in enumerate(zip(rois, cnn_predictions, groups), start=1):
        roi_h = roi["box"][3] - roi["box"][1]

        # Filter vertically distinct boxes in this wheel (vertical offset >= 20% of ROI height)
        distinct_boxes = []
        for det in group:
            if det["yolo_conf"] == 0.0:
                continue
            if not distinct_boxes:
                distinct_boxes.append(det)
            else:
                is_vert_separate = all(abs(det["y"] - existing["y"]) >= (roi_h * 0.20) for existing in distinct_boxes)
                if is_vert_separate:
                    distinct_boxes.append(det)
                else:
                    if det["yolo_conf"] > distinct_boxes[-1]["yolo_conf"]:
                        distinct_boxes[-1] = det

        distinct_boxes = sorted(distinct_boxes, key=lambda d: d["y"])  # Top first, bottom second
        cnn_val = cnn_pred["value"]

        if cnn_val == "N" or len(distinct_boxes) >= 2:
            has_transition = True
            # Rolling transition: bottom is entering digit (d_enter)
            if len(distinct_boxes) >= 2:
                bot_box = distinct_boxes[-1]
                bot_lbl = bot_box["yolo_label"]
                enter_dig = int(bot_lbl) if bot_lbl.isdigit() else 0
                exit_dig = (enter_dig - 1) % 10
            elif distinct_boxes:
                # Deduce from vertical position of single box
                lbl = distinct_boxes[0]["yolo_label"]
                d = int(lbl) if lbl.isdigit() else 0
                center_y = (roi["box"][1] + roi["box"][3]) / 2
                if distinct_boxes[0]["y"] > center_y:
                    enter_dig = d
                    exit_dig = (d - 1) % 10
                else:
                    exit_dig = d
                    enter_dig = (d + 1) % 10
            else:
                enter_dig, exit_dig = 0, 9

            wheel_results.append({
                "position": idx,
                "is_transition": True,
                "display": f"[{exit_dig}->{enter_dig}]",
                "val_before": str(exit_dig),
                "val_after": str(enter_dig),
                "description": f"กำลังหมุน [{exit_dig} -> {enter_dig}]",
                "confidence": cnn_pred["confidence"],
            })
        else:
            # Stable digit from CNN / YOLO Ensemble
            best_yolo = max([d for d in group if d["yolo_conf"] > 0], key=lambda d: d["yolo_conf"], default=None)
            # If YOLO is highly confident on a single box (>= 0.65) and differs from CNN, trust YOLO
            if best_yolo and best_yolo["yolo_conf"] >= 0.65 and str(best_yolo["yolo_label"]) != cnn_val:
                dig = str(best_yolo["yolo_label"])
                conf = best_yolo["yolo_conf"]
            else:
                dig = str(cnn_pred["digit"]) if cnn_pred["digit"] is not None else "?"
                conf = cnn_pred["confidence"]

            wheel_results.append({
                "position": idx,
                "is_transition": False,
                "display": dig,
                "val_before": dig,
                "val_after": dig,
                "description": f"เลขนิ่ง [{dig}]",
                "confidence": conf,
            })

    return wheel_results, has_transition


def save_debug_image(image_path, image, rois, wheel_results):
    canvas = image.copy()
    for roi, res in zip(rois, wheel_results):
        x1, y1, x2, y2 = roi["box"]
        color = (0, 165, 255) if res["is_transition"] else (0, 255, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"#{res['position']} {res['display']}"
        cv2.putText(canvas, text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    output_path = Path("review") / f"cnn_{Path(image_path).name}"
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    return output_path


def read_meter(image_path):
    image_path = Path(image_path)
    cnn_path = Path(CNN_MODEL_PATH)
    if not image_path.exists():
        print(f"ไม่พบไฟล์ภาพ: {image_path}")
        return
    if not cnn_path.exists():
        print(f"ไม่พบ CNN model: {cnn_path}")
        return

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"เปิดภาพไม่ได้: {image_path}")
        return

    detections = get_yolo_detections(image_path)
    groups = cluster_wheel_positions(detections)
    # Complete grid if 1 wheel was missed
    groups = auto_complete_grid(groups, image.shape, EXPECTED_DIGITS)
    rois = make_wheel_rois(image, groups)

    print("\n--- YOLO positions (ใช้หา ROI เท่านั้น) ---")
    for index, group in enumerate(groups, start=1):
        candidates = ", ".join(
            f"{item['yolo_label']}({item['yolo_conf']:.2f})" for item in group if item["yolo_conf"] > 0
        ) or "Auto Grid ROI"
        print(f"ช่อง {index}: {candidates}")

    if len(rois) != EXPECTED_DIGITS:
        print(f"\n⚠️ YOLO พบ {len(rois)} ตำแหน่ง แต่หน้ามิเตอร์ต้องมี {EXPECTED_DIGITS} ช่อง")
        print("หยุดก่อนเพื่อไม่ให้ CNN อ่าน ROI ผิดตำแหน่ง")
        return

    cnn = DigitCNN(cnn_path)
    cnn_predictions = [cnn.predict(roi["image"], CNN_PREPROCESSING) for roi in rois]

    # Resolve using Mechanical Transition Logic
    wheel_results, has_transition = resolve_wheel_readings(rois, cnn_predictions, groups)

    print(f"\n--- ผลการอ่านค่ามิเตอร์ (CNN + Mechanical Transition Logic) ---")
    for res in wheel_results:
        print(f"ช่อง {res['position']}: {res['description']}")

    display_str = "".join(res["display"] for res in wheel_results)
    val_before = "".join(res["val_before"] for res in wheel_results)
    val_after = "".join(res["val_after"] for res in wheel_results)

    print(f"\n📌 ค่าที่อ่านได้: {display_str}")
    if has_transition:
        print(f"📌 ช่วงการเปลี่ยนค่า: {val_before} ➔ {val_after}")
        print(f"📌 ค่าตัวเลขประมาณ: {val_before[:-1]}.{val_before[-1]}")

    if SAVE_DEBUG_IMAGE:
        output_path = save_debug_image(image_path, image, rois, wheel_results)
        print(f"\nบันทึกภาพ ROI สำหรับตรวจสอบ: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("วิธีใช้: python meter_reader3.py path/to/meter_photo.jpg")
        sys.exit(1)
    read_meter(sys.argv[1])

