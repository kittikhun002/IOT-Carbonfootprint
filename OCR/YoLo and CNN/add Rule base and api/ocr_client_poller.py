"""
ocr_client_poller.py
--------------------------------------------------------------------------
รันฝั่ง "เครื่อง OCR" — เชื่อมต่อกับ image-store server อัตโนมัติ:
1. ดึงงานคิวภาพมิเตอร์ที่ยังไม่ได้อ่าน
2. ตรวจสอบประเภทมิเตอร์จากชื่อไฟล์อัตโนมัติ (e... -> elec, w... -> water)
3. ประมวลผลผ่าน Smart Meter Pipeline 4 ขั้นตอน
4. ส่งผลลัพธ์ตัวเลข พร้อมภาพตรวจสอบ (ROI Debug) กลับไปยัง Server
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv

from main_pipeline import run_pipeline, detect_meter_type

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# --- ตั้งค่าการเชื่อมต่อ Server ---------------------------------------------
IMAGE_STORE_BASE_URL = os.getenv("IMAGE_STORE_BASE_URL", "https://localhost:8443")
IMAGE_STORE_VERIFY_TLS = os.getenv("IMAGE_STORE_VERIFY_TLS", "false").lower() == "true"
OCR_SERVICE_USERNAME = os.getenv("OCR_SERVICE_USERNAME", "ocr-service")
OCR_SERVICE_PASSWORD = os.getenv("OCR_SERVICE_PASSWORD", "")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# -----------------------------------------------------------------------

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


def process_one_job(job: dict) -> None:
    job_id = job["id"]
    image_id = job["image_id"]
    filename = job.get("filename", f"{image_id}.jpg")

    # 🏷️ ตรวจประเภทมิเตอร์จากชื่อไฟล์อัตโนมัติ (e -> elec, w -> water)
    meter_type = job.get("meter_type") or detect_meter_type(filename)
    print(f"\n📂 [ocr-client] กำลังประมวลผลงาน #{job_id} (ไฟล์: {filename}, ประเภท: {meter_type})")

    # 1) จองงานนี้ก่อน
    claim_resp = _request("POST", f"/admin/images/ocr/{job_id}/claim")
    if claim_resp.status_code == 409:
        print(f"[ocr-client] งาน #{job_id} ถูกเครื่องอื่นจองไปแล้ว ข้าม", flush=True)
        return
    claim_resp.raise_for_status()

    # 2) โหลดไฟล์ภาพจาก Server
    file_resp = _request("GET", f"/admin/images/{image_id}/file")
    file_resp.raise_for_status()

    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"ocr_job_{job_id}_{filename}"
    tmp_path.write_bytes(file_resp.content)

    try:
        # 3) รันกระบวนการอ่านภาพผ่าน Smart Pipeline
        pipeline_output = run_pipeline(
            image_path=str(tmp_path),
            meter_type=meter_type,
            gemini_key=GEMINI_API_KEY
        )

        status = pipeline_output.get("status")
        
        if status == "APPROVED_LOCAL":
            raw_str = pipeline_output.get("raw", "0")
            reading = float(raw_str) if raw_str.isdigit() else 0.0
            raw_text = f"{pipeline_output.get('reading')} [อนุมัติโดย Local AI]"
        elif status == "APPROVED_GEMINI":
            reading_val = pipeline_output.get("reading", "0")
            reading = float("".join(c for c in reading_val if c.isdigit() or c == "."))
            raw_text = f"{reading_val} [อนุมัติโดย Gemini Vision: {pipeline_output.get('reason', '')}]"
        else:
            errors = pipeline_output.get("local_errors", [])
            raw_text = f"รอตรวจสอบ [ส่งต่อให้คนตรวจ: {'; '.join(errors)}]"
            reading = 0.0

        debug_img_candidate = Path("review") / f"cnn_{tmp_path.name}"
        upload_img_path = debug_img_candidate if debug_img_candidate.exists() else tmp_path

        # 4) ส่งผลการอ่านค่ากลับไปที่ Server
        result_resp = _request(
            "POST", f"/admin/images/ocr/{job_id}/result",
            data={"reading": reading, "raw_text": raw_text},
            files={"result_image": (upload_img_path.name, upload_img_path.read_bytes(), "image/jpeg")},
        )
        result_resp.raise_for_status()
        print(f"✅ [ocr-client] งาน #{job_id} เสร็จสมบูรณ์: reading={reading} ({status})", flush=True)

    except Exception as exc:
        fail_resp = _request(
            "POST", f"/admin/images/ocr/{job_id}/fail",
            json={"error": str(exc)[:500]},
        )
        fail_resp.raise_for_status()
        print(f"❌ [ocr-client] งาน #{job_id} ล้มเหลว: {exc}", flush=True)

    finally:
        tmp_path.unlink(missing_ok=True)


def run_forever() -> None:
    if not OCR_SERVICE_PASSWORD:
        print("⚠️  กรุณาตั้งค่า OCR_SERVICE_PASSWORD ในไฟล์ .env ก่อนเริ่มทำงาน")
        return

    print(f"🚀 [ocr-client] เริ่มทำงาน — ตรวจสอบงานใหม่ทุกๆ {POLL_INTERVAL_SECONDS} วินาที...", flush=True)
    while True:
        try:
            resp = _request("GET", "/admin/images/ocr", params={"job_status": "queued", "limit": BATCH_SIZE})
            resp.raise_for_status()
            jobs = resp.json()
            if jobs:
                print(f"\n📦 พบงานใหม่ {len(jobs)} รายการ กำลังประมวลผล...")
            for job in jobs:
                process_one_job(job)
        except Exception as exc:
            print(f"[ocr-client] ข้อผิดพลาดขณะดึงงาน: {exc}", flush=True)
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()