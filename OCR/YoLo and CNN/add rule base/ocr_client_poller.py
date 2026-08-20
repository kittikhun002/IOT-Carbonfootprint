"""
ocr_client_poller.py
--------------------------------------------------------------------------
รันฝั่ง "เครื่อง OCR" — แทนที่ ocr_api.py (ไม่ต้องมี web server ฝั่ง OCR
อีกต่อไปในโหมดนี้) กลับบทบาทเป็น: OCR ไปถาม/จอง/ประมวลผล/ส่งผลกลับหา
image-store เอง เป็นระยะๆ (async pull) แทนที่ image-store จะยิงภาพมาหา

ใช้ logic การอ่านค่าเดิมจาก meter_reader.py ทุกจุด (YOLO + กรองเลขซ้อน)
ไม่ได้แก้อะไรในไฟล์นั้นเลย — วางไฟล์นี้ไว้โฟลเดอร์เดียวกับ meter_reader.py,
best.pt, data.yaml

หลังอ่านค่าสำเร็จ จะส่งภาพต้นฉบับที่ดาวน์โหลดมา (ไฟล์เดียวกับที่ประมวลผล
ไม่ได้แก้ไข/annotate อะไรเพิ่ม) แนบกลับไปพร้อมผลลัพธ์ด้วย — image-store จะ
เก็บไว้แยกเป็น "ภาพจาก OCR" คู่กับภาพต้นฉบับ ให้ดูเทียบกันได้ในหน้า
dashboard

--------------------------------------------------------------------------
⚠️ สำคัญ: ใช้โหมดนี้ "แทนที่" ocr-worker ของ image-store ไม่ใช่ใช้คู่กัน
   ถ้ารันทั้งสองอย่างพร้อมกัน จะมีโอกาสประมวลผลภาพเดียวกันซ้ำ 2 รอบ —
   ต้องสั่งหยุด ocr-worker container ก่อน:
       cd image-store && docker compose stop ocr-worker
--------------------------------------------------------------------------

SETUP (ทำครั้งเดียว):

  1. สร้างบัญชี "OCR service account" ที่ image-store (เป็น admin เพราะ
     endpoint ที่ใช้ทั้งหมดอยู่ใต้ /admin/):

       curl -sk -X POST https://<IMAGE_STORE_HOST>:8443/register \\
         -H "Content-Type: application/json" \\
         -d '{"username":"ocr-service","password":"<ตั้งรหัสเอง>"}'

     แล้ว promote เป็น admin (และ mark เป็น device account กันไม่ให้ไป
     โผล่ในลิสต์ user จริงในหน้า dashboard — ดูที่คุยกันไว้เรื่อง esp32):

       docker compose exec db psql -U imguser -d imagestore -c "
         UPDATE users SET is_admin = true, is_device = true
         WHERE username = 'ocr-service';"

  2. ติดตั้ง python packages (เพิ่มจากที่ meter_reader.py ต้องใช้อยู่แล้ว):
       pip install httpx python-dotenv ultralytics opencv-python pytesseract --break-system-packages

  3. สร้างไฟล์ .env ในโฟลเดอร์เดียวกับสคริปต์นี้ (ดู .env.example ที่แนบมา
     ด้วย) แล้วรัน:
       python3 ocr_client_poller.py

     โหลดค่าจาก .env อัตโนมัติทุกครั้งที่รัน — ไม่ต้อง export เองทีละตัว
"""
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from ultralytics import YOLO

import meter_reader  # ใช้ MODEL_PATH, CONFIDENCE_THRESHOLD จากไฟล์เดิมตรงๆ

load_dotenv()  # อ่านไฟล์ .env ที่อยู่โฟลเดอร์เดียวกัน (ถ้ามี) ก่อนอ่านค่า config ด้านล่าง

# --- ตั้งค่า -----------------------------------------------------------
IMAGE_STORE_BASE_URL = os.getenv("IMAGE_STORE_BASE_URL", "https://localhost:8443")
IMAGE_STORE_VERIFY_TLS = os.getenv("IMAGE_STORE_VERIFY_TLS", "false").lower() == "true"
OCR_SERVICE_USERNAME = os.getenv("OCR_SERVICE_USERNAME", "ocr-service")
OCR_SERVICE_PASSWORD = os.getenv("OCR_SERVICE_PASSWORD", "")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
# -----------------------------------------------------------------------

_model = YOLO(meter_reader.MODEL_PATH)  # โหลดครั้งเดียวตอนสคริปต์เริ่ม
_token: str | None = None


def _login() -> str:
    global _token
    resp = httpx.post(
        f"{IMAGE_STORE_BASE_URL}/login",
        json={"username": OCR_SERVICE_USERNAME, "password": OCR_SERVICE_PASSWORD},
        verify=IMAGE_STORE_VERIFY_TLS,
        timeout=10.0,
    )
    resp.raise_for_status()
    _token = resp.json()["access_token"]
    return _token


def _auth_headers() -> dict:
    if _token is None:
        _login()
    return {"Authorization": f"Bearer {_token}"}


def _request(method: str, path: str, retry: bool = True, **kwargs) -> httpx.Response:
    resp = httpx.request(
        method,
        f"{IMAGE_STORE_BASE_URL}{path}",
        headers=_auth_headers(),
        verify=IMAGE_STORE_VERIFY_TLS,
        timeout=30.0,
        **kwargs,
    )
    if resp.status_code == 401 and retry:
        global _token
        _token = None
        return _request(method, path, retry=False, **kwargs)
    return resp


def read_digits(image_path: Path) -> list[dict]:
    """เหมือน meter_reader.read_digits_with_yolo() ทุกจุด (รวมกรองเลขซ้อน
    ในช่องเดียวกัน) ต่างแค่ใช้โมเดลที่โหลดไว้แล้วครั้งเดียว (_model)"""
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


def process_one_job(job: dict) -> None:
    job_id = job["id"]
    image_id = job["image_id"]

    # 1) จองงานนี้ก่อน (กันคนอื่น/รอบอื่นมาแย่งประมวลผลซ้ำ)
    claim_resp = _request("POST", f"/admin/images/ocr/{job_id}/claim")
    if claim_resp.status_code == 409:
        print(f"[ocr-client] job {job_id} ถูกจองไปแล้ว ข้าม", flush=True)
        return
    claim_resp.raise_for_status()

    # 2) โหลดไฟล์ภาพมาไว้ในเครื่องชั่วคราว
    file_resp = _request("GET", f"/admin/images/{image_id}/file")
    file_resp.raise_for_status()

    tmp_path = Path(f"/tmp/ocr_job_{job_id}.jpg")
    tmp_path.write_bytes(file_resp.content)

    try:
        detections = read_digits(tmp_path)
        if not detections:
            raise ValueError("YOLO ไม่พบตัวเลขในภาพเลย")

        reading_str = "".join(d["label"] for d in detections)
        min_conf = min(d["conf"] for d in detections)
        reading = float(reading_str)

        raw_text = f"{reading_str} (digits={len(detections)}, min_conf={min_conf:.2f})"
        if min_conf < meter_reader.CONFIDENCE_THRESHOLD:
            raw_text += " [LOW CONFIDENCE - ควรตรวจสอบด้วยตา]"

        # 3a) สำเร็จ — ส่งผลกลับ พร้อมแนบภาพเดิมที่ดาวน์โหลดมากลับไปด้วย
        # (ใช้ data=/files= เพราะฝั่ง endpoint นี้รับแบบ multipart ไม่ใช่ JSON
        # แล้ว — เพื่อให้แนบไฟล์ภาพไปพร้อมกันได้ในคำขอเดียว)
        result_resp = _request(
            "POST", f"/admin/images/ocr/{job_id}/result",
            data={"reading": reading, "raw_text": raw_text},
            files={"result_image": (tmp_path.name, tmp_path.read_bytes(), "image/jpeg")},
        )
        result_resp.raise_for_status()
        print(f"[ocr-client] job {job_id} เสร็จ: reading={reading}", flush=True)

    except Exception as exc:
        # 3b) ล้มเหลว — รายงาน error กลับไป ให้รอบถัดไป retry ให้เอง
        fail_resp = _request(
            "POST", f"/admin/images/ocr/{job_id}/fail",
            json={"error": str(exc)[:500]},
        )
        fail_resp.raise_for_status()
        print(f"[ocr-client] job {job_id} ล้มเหลว: {exc}", flush=True)

    finally:
        tmp_path.unlink(missing_ok=True)


def run_forever() -> None:
    if not OCR_SERVICE_PASSWORD:
        raise RuntimeError("ตั้งค่า OCR_SERVICE_PASSWORD ก่อน (ดูวิธีสร้างบัญชีที่คอมเมนต์บนสุดของไฟล์นี้)")

    print(f"[ocr-client] เริ่มทำงาน — poll ทุก {POLL_INTERVAL_SECONDS} วิ, batch {BATCH_SIZE}", flush=True)
    while True:
        try:
            resp = _request("GET", "/admin/images/ocr", params={"job_status": "queued", "limit": BATCH_SIZE})
            resp.raise_for_status()
            jobs = resp.json()
            for job in jobs:
                process_one_job(job)
        except Exception as exc:
            print(f"[ocr-client] poll error: {exc}", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()