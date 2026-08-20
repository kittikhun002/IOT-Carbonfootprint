"""
ocr_api.py
--------------------------------------------------------------------------
ห่อ meter_reader.py เดิม (YOLO + pytesseract) ให้เป็น web API ที่
image-store ยิงมาหาได้ตรงตาม contract (ดู call_ocr_api() ใน
image-store/app/ocr_worker.py) — ไม่แก้ logic การอ่านค่าใน
meter_reader.py เลยแม้แต่บรรทัดเดียว แค่เพิ่มชั้น HTTP ครอบไว้ข้างนอก

วางไฟล์นี้ไว้ในโฟลเดอร์เดียวกับ meter_reader.py, best.pt, data.yaml

รับ:
    POST /ocr
    Content-Type: multipart/form-data
    Authorization: Bearer <OCR_API_KEY>   (ถ้าตั้ง OCR_API_KEY ไว้)
    fields: image (ไฟล์ภาพ), meter_id, captured_at

ตอบกลับ:
    {"reading": 12345.0, "raw_text": "12345 (digits=5, min_conf=0.91)"}

ติดตั้งเพิ่ม (นอกจากที่ meter_reader.py ต้องใช้อยู่แล้ว — ultralytics,
opencv-python, pytesseract):
    pip install fastapi uvicorn python-multipart --break-system-packages

รัน (ต้องรันจากในโฟลเดอร์นี้ เพราะ MODEL_PATH="best.pt" เป็น relative path):
    cd ocr/
    uvicorn ocr_api:app --host 0.0.0.0 --port 8000

โมเดลโหลดครั้งเดียวตอน service เริ่มทำงาน ไม่ใช่ทุก request — เร็วกว่าเรียก
meter_reader.py ผ่าน subprocess ทีละภาพมาก (โหลด best.pt ใหม่ทุกครั้งช้ามาก)
"""
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from ultralytics import YOLO

import meter_reader  # ใช้ค่าคงที่ + logic เดิมจากไฟล์นี้ตรงๆ ไม่ก็อปโค้ดซ้ำ

app = FastAPI(title="Meter OCR API")

# ต้องตั้งค่าตัวเดียวกันทั้ง 2 ฝั่ง — ที่นี่ (OCR_API_KEY) และใน
# image-store/.env (ก็ชื่อ OCR_API_KEY เหมือนกัน) ปล่อยว่างได้ถ้าไม่ต้องการ
# auth (เช่นตอนทดสอบในวง LAN ปิด)
OCR_API_KEY = os.getenv("OCR_API_KEY", "")

# โหลดโมเดลครั้งเดียวตอน startup
_model = YOLO(meter_reader.MODEL_PATH)


def read_digits(image_path: Path) -> list[dict]:
    """เหมือน meter_reader.read_digits_with_yolo() ทุกจุด (กรองเลขซ้อนในช่อง
    เดียวกันแบบเดียวกันด้วย) ต่างแค่ใช้โมเดลที่โหลดไว้แล้ว (_model) แทนการ
    โหลดไฟล์ .pt ใหม่ทุก request"""
    results = _model(str(image_path), iou=0.3, agnostic_nms=True, conf=0.15)

    detections = []
    for r in results:
        class_names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            if label == "Reading Digit":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            center_x = (x1 + x2) / 2
            detections.append({"label": label, "conf": conf, "x": center_x})

    detections.sort(key=lambda d: d["x"])

    filtered: list[dict] = []
    X_PROXIMITY_THRESHOLD = 30.0
    for d in detections:
        if not filtered:
            filtered.append(d)
            continue
        if abs(d["x"] - filtered[-1]["x"]) < X_PROXIMITY_THRESHOLD:
            if d["conf"] > filtered[-1]["conf"]:
                filtered[-1] = d
        else:
            filtered.append(d)

    return filtered


@app.post("/ocr")
async def ocr(
    image: UploadFile = File(...),
    meter_id: str = Form(...),
    captured_at: str = Form(...),
    authorization: str | None = Header(None),
):
    if OCR_API_KEY:
        if authorization != f"Bearer {OCR_API_KEY}":
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    image_bytes = await image.read()

    # YOLO ในนี้ต้องการ path ไฟล์ (เหมือน meter_reader.py เดิม) เลยเขียนลง
    # ไฟล์ชั่วคราวก่อน แล้วลบทิ้งทันทีหลังใช้เสร็จ
    suffix = Path(image.filename or "photo.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = Path(tmp.name)

    try:
        detections = read_digits(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not detections:
        # 4xx/5xx อะไรก็ได้ที่ไม่ใช่ 200 — ฝั่ง ocr_worker.py จะ mark เป็น
        # failed แล้ว retry เองตามรอบ ไม่ต้องจัดการ retry ซ้อนฝั่งนี้
        raise HTTPException(status_code=422, detail="YOLO ไม่พบตัวเลขในภาพเลย")

    reading_str = "".join(d["label"] for d in detections)
    min_conf = min(d["conf"] for d in detections)

    try:
        reading = float(reading_str)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"อ่านค่าไม่เป็นตัวเลข: '{reading_str}'")

    raw_text = f"{reading_str} (digits={len(detections)}, min_conf={min_conf:.2f})"
    if min_conf < meter_reader.CONFIDENCE_THRESHOLD:
        raw_text += " [LOW CONFIDENCE - ควรตรวจสอบด้วยตา]"

    return {"reading": reading, "raw_text": raw_text}


@app.get("/health")
def health():
    return {"status": "ok"}
