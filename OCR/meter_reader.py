"""
Meter Reader Prototype — YOLO (detection) + Tesseract (OCR)
100% free, runs entirely on your own machine, no API calls.

=========================================================
SETUP (ทำครั้งเดียว)
=========================================================

1. ติดตั้ง Tesseract engine (ตัวโปรแกรม ไม่ใช่ python package):
   - Windows: โหลดตัวติดตั้งจาก https://github.com/UB-Mannheim/tesseract/wiki
   - Mac:     brew install tesseract
   - Linux:   sudo apt install tesseract-ocr

2. ติดตั้ง python packages:
   pip install ultralytics pytesseract opencv-python pillow --break-system-packages

3. (Windows เท่านั้น) ถ้า tesseract ไม่อยู่ใน PATH ให้ตั้งค่า path เอง
   ในโค้ดข้างล่าง หา TESSERACT_CMD แล้วแก้ path ให้ตรงกับที่ติดตั้งจริง

4. หาโมเดล YOLO ที่ fine-tune มาสำหรับมิเตอร์แล้ว (ฟรี):
   - ค้นบน Roboflow Universe: https://universe.roboflow.com
     คำค้น: "digital meter reading" หรือ "seven segment display"
   - หรือค้น GitHub: "meter reading YOLO weights"
   - โหลดไฟล์ .pt มาแล้ววางไว้ในโฟลเดอร์เดียวกับสคริปต์นี้
   - ตั้งชื่อไฟล์เป็น meter_digits.pt หรือแก้ MODEL_PATH ข้างล่างให้ตรง

   ** ถ้ายังไม่มีโมเดล fine-tune — สคริปต์นี้ยังใช้ได้ โดยข้าม YOLO ไปเลย
      (ตั้ง USE_YOLO = False) แล้วให้ Tesseract อ่านทั้งภาพ/crop มือเอง
      ความแม่นจะต่ำกว่า แต่ใช้ทดสอบไอเดียได้ก่อน

=========================================================
วิธีใช้
=========================================================
python meter_reader.py path/to/meter_photo.jpg
"""

import sys
import cv2
import pytesseract
from pathlib import Path

# ---------- ตั้งค่าตรงนี้ ----------
USE_YOLO = True                       # False = ข้าม YOLO ใช้ Tesseract อ่านทั้งภาพ
MODEL_PATH = "best.pt"                # path ไปยังไฟล์ weight ที่เทรนมาจาก Colab
CONFIDENCE_THRESHOLD = 0.85           # ต่ำกว่านี้ = ต้องให้คนตรวจสอบ
TESSERACT_CMD = None                  # Windows: ใส่ path เช่น r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# -----------------------------------

if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def preprocess_for_ocr(crop):
    """แปลงภาพให้ Tesseract อ่านง่ายขึ้น: grayscale -> threshold (ใช้เฉพาะโหมด fallback)"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def ocr_digits(crop):
    """อ่านตัวเลขจาก crop ด้วย Tesseract (fallback โหมดถ้าไม่ใช้ YOLO)"""
    processed = preprocess_for_ocr(crop)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed, config=config)
    return text.strip()


def read_digits_with_yolo(image_path):
    """
    ใช้ YOLO ที่ fine-tune มาแล้ว อ่านตัวเลขโดยตรง (ไม่ต้องพึ่ง Tesseract อีกขั้น)
    โมเดลนี้มี 11 classes: '0'-'9' (ตัวเลข) และ 'Reading Digit' (กรอบรวม ไม่ใช้ในการอ่านค่า)
    """
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    # iou ต่ำ = กรองกรอบซ้อนทับกันเข้มขึ้น, agnostic_nms=True = ตัดกรอบซ้อนแม้ต่าง class กัน
    # conf=0.15 = ลด threshold ให้เจอกรอบที่ไม่มั่นใจด้วย (ปกติ YOLO default กรองทิ้งที่ 0.25)
    # ค่า confidence ที่ต่ำจะยังโดน flag ให้ตรวจสอบอยู่ดีผ่าน CONFIDENCE_THRESHOLD ข้างล่าง
    results = model(image_path, iou=0.3, agnostic_nms=True, conf=0.15)

    img = cv2.imread(str(image_path))
    detections = []

    for r in results:
        class_names = r.names  # dict เช่น {0: '0', 1: '1', ..., 10: 'Reading Digit'}
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]

            # ข้าม class 'Reading Digit' เพราะเป็นแค่กรอบรวม ไม่ใช่ตัวเลขจริง
            if label == "Reading Digit":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            center_x = (x1 + x2) / 2  # ใช้จุดกึ่งกลางแทนขอบซ้าย แม่นกว่าเวลากรอบกว้างไม่เท่ากัน
            detections.append({"label": label, "conf": conf, "x": center_x, "box": (x1, y1, x2, y2)})

    # เรียงจากซ้ายไปขวา ตามตำแหน่งจริงบนมิเตอร์ (ใช้จุดกึ่งกลางกรอบ)
    detections.sort(key=lambda d: d["x"])

    # DEBUG: แสดงตำแหน่งจริงที่ตรวจเจอ เพื่อเช็คว่าทำไมลำดับผิด
    print("\n--- DEBUG: รายละเอียดกรอบที่ YOLO เจอ (เรียงแล้ว) ---")
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        print(f"label='{d['label']}'  center_x={d['x']:.1f}  conf={d['conf']:.2f}  box=(x1={x1}, y1={y1}, x2={x2}, y2={y2})")
    print("--- END DEBUG ---\n")

    # เช็คช่องว่างระหว่างตัวเลข — ถ้าห่างกันผิดปกติ (เกิน 1.6 เท่าของระยะห่างเฉลี่ย) อาจมีตัวเลขที่ตรวจไม่เจอแทรกอยู่
    if len(detections) >= 3:
        gaps = [detections[i+1]["x"] - detections[i]["x"] for i in range(len(detections)-1)]
        avg_gap = sum(gaps) / len(gaps)
        for i, gap in enumerate(gaps):
            if gap > avg_gap * 1.6:
                print(f"⚠️  ช่องว่างระหว่าง '{detections[i]['label']}' กับ '{detections[i+1]['label']}' "
                      f"กว้างผิดปกติ ({gap:.0f}px เทียบเฉลี่ย {avg_gap:.0f}px) — อาจมีตัวเลขที่ตรวจไม่เจอแทรกอยู่ตรงนี้")

    return detections


def read_meter(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ไม่พบไฟล์: {image_path}")
        return

    if USE_YOLO:
        detections = read_digits_with_yolo(image_path)
        if not detections:
            print("YOLO ไม่พบตัวเลขในภาพเลย — เช็คว่า MODEL_PATH ถูกต้อง หรือลองภาพอื่น")
            return

        reading = "".join(d["label"] for d in detections)
        min_conf = min(d["conf"] for d in detections)

        print(f"ค่าที่อ่านได้: {reading}")
        print(f"จำนวนหลักที่ตรวจพบ: {len(detections)}")
        print(f"Confidence ต่ำสุดในกลุ่ม: {min_conf:.2f}")
        print("รายละเอียดแต่ละหลัก:")
        for d in detections:
            print(f"  หลัก '{d['label']}' — confidence {d['conf']:.2f}")

        if min_conf < CONFIDENCE_THRESHOLD:
            print("⚠️  Confidence ต่ำกว่า threshold — แนะนำให้ตรวจสอบด้วยตาอีกครั้ง")
        else:
            print("✅ Confidence สูงพอ — ใช้ค่านี้ได้")

    else:
        img = cv2.imread(str(image_path))
        text = ocr_digits(img)
        print(f"ค่าที่อ่านได้ (ไม่ผ่าน YOLO ครอปก่อน): {text if text else '(อ่านไม่ได้)'}")
        print("หมายเหตุ: ไม่ crop ก่อน ความแม่นจะต่ำกว่ามาก แนะนำครอปภาพเฉพาะช่องตัวเลขด้วยมือก่อนถ้าจะใช้โหมดนี้")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("วิธีใช้: python meter_reader.py path/to/meter_photo.jpg")
        sys.exit(1)

    read_meter(sys.argv[1])