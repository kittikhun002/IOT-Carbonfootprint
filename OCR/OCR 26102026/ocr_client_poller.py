"""
ocr_client_poller.py
--------------------------------------------------------------------------
ระบบ OCR Client Worker — เชื่อมต่อกับ Image Store Server (https://cfo.ntplc.co.th/iot)
ตาม Flow การทำงาน 7 ขั้นตอน (พร้อมระบบแยก Test / Production):

1. สอบถามคิวงาน (Poll Job Queue)            -> GET  /admin/images/ocr?job_status=queued
2. จองงานและรับชุดภาพ (Claim Job)           -> POST /admin/images/ocr/{job_id}/claim
3. ตรวจสอบสถานะการทดสอบ (Check Test Flag)   -> ตรวจคำว่า "test" ในชื่อไฟล์/URL (is_test = True/False)
4. ดาวน์โหลดรูปภาพแบบ Dynamic (Download)    -> โหลดทุกภาพในกลุ่มเข้า downloads/ ไม่จำกัดขั้นต่ำ
5. จัดการประวัติมิเตอร์ตามเงื่อนไข (History)  -> ถ้า is_test=True ข้ามประวัติ (history=[]), ถ้า False ดึงปกติ
6. ประมวลผลภาพด้วย AI (Run OCR Pipeline)    -> YOLO + CNN + 3 Rules + Majority Vote (2 ใน 3) + Gemini Fallback
7. แยก Endpoint ส่งผลลัพธ์ (Submit Result)  -> ส่ง error_type, ocr_reading และ ocr_engine ไปยังตาราง Test หรือ Production
"""

import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from main_pipeline import run_multi_image_pipeline, detect_meter_type

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ===================================================================
# 🔧 ตั้งค่าการเชื่อมต่อ Server (แก้ได้ใน .env)
# ===================================================================
IMAGE_STORE_BASE_URL = os.getenv("IMAGE_STORE_BASE_URL", "https://cfo.ntplc.co.th/iot").rstrip("/")
IMAGE_STORE_VERIFY_TLS = os.getenv("IMAGE_STORE_VERIFY_TLS", "true").lower() == "true"
OCR_API_KEY = os.getenv("OCR_API_KEY", "").strip()
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "20"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# ===================================================================
# 🛣️ API Paths
# ===================================================================
API_GET_JOBS            = "/admin/images/ocr"                             # GET  — poll หา job ที่ job_status=queued
API_CLAIM_JOB           = "/admin/images/ocr/{job_id}/claim"              # POST — claim job -> ได้ image_file_urls
API_GET_IMAGE_FILE      = "/admin/images/{item_id}/file"                  # GET  — โหลดไฟล์ภาพ
API_GET_METER_READINGS  = "/admin/meters/{meter_id}/ocr-readings"         # GET  — ดึงประวัติ ocr_meter ย้อนหลัง
API_SUBMIT_RESULT       = "/admin/images/ocr/{job_id}/result"             # POST — ส่งผลลัพธ์ Production
API_SUBMIT_RESULT_TEST  = os.getenv("API_SUBMIT_RESULT_TEST", "/admin/images/ocr/{job_id}/result-test") # POST — ส่งผลลัพธ์ Test
API_SUBMIT_FAIL         = "/admin/images/ocr/{job_id}/fail"               # POST — แจ้ง Technical/Network Error
# ===================================================================


def _auth_headers() -> dict:
    """Header ยืนยันตัวตนด้วย X-OCR-Key"""
    headers = {}
    if OCR_API_KEY:
        headers["X-OCR-Key"] = OCR_API_KEY
    return headers


def _request(method: str, path_or_url: str, **kwargs) -> httpx.Response:
    """ส่ง HTTP Request ไปยัง Image Store Server"""
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        url = path_or_url
    else:
        clean_path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        url = f"{IMAGE_STORE_BASE_URL}{clean_path}"

    headers = kwargs.pop("headers", {})
    headers.update(_auth_headers())

    return httpx.request(
        method,
        url,
        headers=headers,
        verify=IMAGE_STORE_VERIFY_TLS,
        timeout=30.0,
        **kwargs,
    )


# -------------------------------------------------------------------
# 📦 Step 1: สอบถามคิวงาน (Poll Job Queue)
# -------------------------------------------------------------------
def fetch_queued_jobs() -> list[dict]:
    """ดึงรายการ job ทั้งหมดที่มีสถานะ queued"""
    try:
        resp = _request("GET", API_GET_JOBS, params={"job_status": "queued", "limit": BATCH_SIZE})
        resp.raise_for_status()
        jobs = resp.json()
        return jobs if isinstance(jobs, list) else []
    except Exception as exc:
        print(f"⚠️ [Step 1] ดึงคิวงานไม่สำเร็จ: {exc}", flush=True)
        return []


# -------------------------------------------------------------------
# 🔒 Step 2 & 4: Claim จองงาน และดาวน์โหลดภาพแบบ Dynamic
# -------------------------------------------------------------------
def claim_and_download_images(job: dict) -> tuple[bool, list[str], list[str]]:
    """
    Step 2: สั่ง Claim งานผ่าน POST /admin/images/ocr/{job_id}/claim
            เพื่อรับ image_file_urls ทั้งหมดของกลุ่ม
    Step 4: ดาวน์โหลดภาพแบบ Dynamic ตามจำนวน URL ที่ได้มาเข้า downloads/
    """
    job_id = job["id"]
    download_dir = Path("downloads")
    download_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[str] = []

    try:
        # Step 2: Claim Job
        claim_resp = _request("POST", API_CLAIM_JOB.format(job_id=job_id))
        if claim_resp.status_code == 409:
            print(f"   ⚠️ งาน #{job_id} ถูกเครื่องอื่น Claim ไปแล้ว ข้าม", flush=True)
            return False, [], []
        claim_resp.raise_for_status()
        claim_data = claim_resp.json()

        image_urls = claim_data.get("image_file_urls", [])
        if not image_urls:
            image_id = job.get("image_id")
            if image_id:
                image_urls = [API_GET_IMAGE_FILE.format(item_id=image_id)]

        print(f"   🔒 Claim สำเร็จ: งาน #{job_id} (พบ {len(image_urls)} ภาพในกลุ่ม)", flush=True)

        # Step 4: ดาวน์โหลดภาพแบบ Dynamic ตามจำนวนจริงที่มี
        for idx, img_url in enumerate(image_urls, start=1):
            file_resp = _request("GET", img_url)
            file_resp.raise_for_status()

            filename = f"job{job_id}_shot{idx}.jpg"
            img_path = download_dir / f"ocr_{filename}"
            img_path.write_bytes(file_resp.content)
            image_paths.append(str(img_path))
            print(f"      📥 ดาวน์โหลดรูปที่ {idx}: {img_path.name} ({len(file_resp.content)/1024:.1f} KB)", flush=True)

        return True, image_paths, image_urls

    except Exception as exc:
        print(f"   ❌ Claim หรือดาวน์โหลดภาพล้มเหลว: {exc}", flush=True)
        return False, [], []


# -------------------------------------------------------------------
# 📊 Step 5: จัดการประวัติมิเตอร์ (เฉพาะโหมดจริง)
# -------------------------------------------------------------------
def fetch_meter_history(meter_id: str) -> list[float]:
    """
    ดึงประวัติการอ่านที่สำเร็จจาก GET /admin/meters/{meter_id}/ocr-readings?only_successful=true
    """
    try:
        resp = _request(
            "GET",
            API_GET_METER_READINGS.format(meter_id=meter_id),
            params={"limit": 3, "only_successful": "true"}
        )
        if resp.status_code == 200:
            entries = resp.json()
            history = []
            if isinstance(entries, list):
                # เรียงลำดับจากเก่าไปใหม่ เพื่อให้ history[-1] คือค่าเดือนล่าสุดเสมอ
                try:
                    entries = sorted(entries, key=lambda x: (x.get("created_at") or x.get("id") or 0) if isinstance(x, dict) else 0)
                except Exception:
                    pass

                for entry in entries:
                    if isinstance(entry, (int, float)):
                        history.append(float(entry))
                        continue
                    reading_val = entry.get("ocr_reading")
                    err = entry.get("error_type")
                    if reading_val is not None and (err is None or err == 0):
                        try:
                            history.append(float(reading_val))
                        except (ValueError, TypeError):
                            pass
            print(f"   📊 ประวัติอ่านสำเร็จย้อนหลัง (เก่า->ใหม่): {history}", flush=True)
            return history
        else:
            print(f"   ⚠️ ไม่พบประวัติเดิม (HTTP {resp.status_code}) — ข้ามไปทำต่อ", flush=True)
            return []
    except Exception as exc:
        print(f"   ⚠️ ดึงประวัติ Error: {exc} — ข้ามไปทำต่อ", flush=True)
        return []


# -------------------------------------------------------------------
# 🧠 Main Flow: ประมวลผล 1 Job ตาม 7 ขั้นตอน
# -------------------------------------------------------------------
def process_single_job(job: dict) -> None:
    job_id = job["id"]
    original_filename = str(job.get("original_filename", ""))
    meter_id = job.get("meter_id") or Path(original_filename).stem

    print(f"\n{'='*65}")
    print(f"🔍 [กำลังประมวลผล] งาน #{job_id} | มิเตอร์: '{meter_id}' | ไฟล์: '{original_filename}'")
    print(f"{'='*65}")

    # --- Step 2 & 4: Claim และดาวน์โหลดภาพแบบ Dynamic ---
    success, image_paths, image_urls = claim_and_download_images(job)
    if not success or not image_paths:
        return

    # --- Step 3: ตรวจสอบสถานะการทดสอบ (Check Test Flag) ---
    is_test = "test" in original_filename.lower() or any("test" in str(u).lower() for u in image_urls)
    if is_test:
        print(f"   🏷️ โหมดการทำงาน: 🧪 TEST (ตรวจพบคำว่า 'test' -> ข้ามประวัติ / error_type 0-2)", flush=True)
    else:
        print(f"   🏷️ โหมดการทำงาน: 🏢 PRODUCTION (งานจริง -> ตรวจครบ 3 กฎ / error_type 0-3)", flush=True)

    # --- Step 5: จัดการประวัติมิเตอร์ตามเงื่อนไข (Conditional History) ---
    if is_test:
        history = []
        print("   ⏩ [is_test=True] ข้ามการดึงประวัติมิเตอร์ (history = [])", flush=True)
    else:
        history = fetch_meter_history(meter_id)

    # ตรวจจับประเภทมิเตอร์ (elec, water, gas)
    meter_type = detect_meter_type(original_filename)
    if meter_type == "auto" and meter_id:
        meter_type = detect_meter_type(meter_id)

    try:
        # --- Step 6: ประมวลผลภาพด้วย AI (Run OCR Pipeline + Majority Vote) ---
        print(f"\n[Step 6] 🧠 กำลังอ่านภาพ AI (YOLO + CNN) + Majority Vote ({len(image_paths)} รูป)...")

        pipeline_output = run_multi_image_pipeline(
            image_paths=image_paths,
            meter_type=meter_type,
            history=history,
            gemini_key=GEMINI_API_KEY,
        )

        status = pipeline_output.get("status")
        local_errors = pipeline_output.get("local_errors", [])
        raw_str = pipeline_output.get("raw") or ""
        reading_str = pipeline_output.get("reading") or ""

        # ดึงตัวเลขพร้อมจุดทศนิยม
        reading_parts = reading_str.strip().split() if reading_str else []
        if reading_parts:
            digits_only = "".join(c for c in reading_parts[0] if c.isdigit() or c == ".")
        else:
            digits_only = "".join(c for c in str(raw_str) if c.isdigit() or c == ".")

        # -------------------------------------------------------------------
        # 🏷️ แมป error_type:
        # 0 = สำเร็จ (Success)
        # 1 = ภาพอ่านไม่ออก / ความมั่นใจต่ำ / เฟืองขัดแย้ง (image_unreadable)
        # 2 = ตรวจไม่พบตัวเลขในภาพ (no_digits_found)
        # 3 = ค่ามิเตอร์ลดลง หรือ การใช้พุ่งผิดปกติ (เฉพาะ Production)
        # -------------------------------------------------------------------
        error_type: int = 0
        ocr_reading: float | None = None

        if status in ["APPROVED_LOCAL", "APPROVED_GEMINI"] and digits_only:
            # ✅ เคส 0: สำเร็จ
            error_type = 0
            try:
                ocr_reading = float(digits_only)
            except ValueError:
                try:
                    cleaned = digits_only.replace(".", "", digits_only.count(".") - 1)
                    ocr_reading = float(cleaned)
                except (ValueError, TypeError):
                    error_type = 1
                    ocr_reading = None
        else:
            # ❌ เกิดข้อผิดพลาด
            joined_errs = " ".join(local_errors).lower()

            if "ไม่พบล้อตัวเลข" in joined_errs or "no digits" in joined_errs:
                # เคส 2: no_digits_found (ไม่ส่ง ocr_reading)
                error_type = 2
                ocr_reading = None

            elif not is_test and ("ลดลง" in joined_errs or "decreased" in joined_errs or "ผิดปกติ" in joined_errs or "anomaly" in joined_errs or "พุ่งสูง" in joined_errs):
                # เคส 3: reading_decreased / usage_anomaly (เฉพาะโหมด Production)
                error_type = 3
                try:
                    ocr_reading = float(digits_only) if digits_only else None
                except ValueError:
                    ocr_reading = None

            else:
                # เคส 1: image_unreadable (ไม่ส่ง ocr_reading)
                error_type = 1
                ocr_reading = None

        # --- Step 7: แยก Endpoint ส่งผลลัพธ์ (Submit Result by Flag) ---
        print(f"\n[Step 7] 📤 กำลังส่งผลลัพธ์กลับ Server...")

        # ดึงค่า ocr_engine จาก Pipeline (Summation Score: 0, 120, 1120, 110, 1110)
        ocr_engine: int = int(pipeline_output.get("ocr_engine", 0 if status == "APPROVED_LOCAL" else (120 if status == "APPROVED_GEMINI" else 110)))

        # ดึงค่า confidence จาก Pipeline (แปลงเป็นสเกล 0-100 float ตามข้อกำหนด Server)
        conf_raw = pipeline_output.get("confidence")
        confidence_pct: float | None = None
        if conf_raw is not None:
            c = float(conf_raw)
            if c <= 1.0:
                c *= 100.0
            confidence_pct = round(max(0.0, min(100.0, c)), 2)

        form_data = {
            "error_type": error_type,
            "ocr_engine": ocr_engine,
        }
        # ocr_reading ต้องเป็นค่า >= 0 ตามข้อกำหนด Server (กรณี error_type 0, 3)
        if ocr_reading is not None and ocr_reading >= 0 and error_type in (0, 3):
            form_data["ocr_reading"] = digits_only
            if confidence_pct is not None:
                form_data["confidence"] = confidence_pct

        # เลือก Endpoint ปลายทางตามสถานะ is_test
        if is_test:
            target_endpoint = API_SUBMIT_RESULT_TEST.format(job_id=job_id)
            print(f"   🧪 [Test Mode] ส่งไปยัง Endpoint: {target_endpoint}", flush=True)
        else:
            target_endpoint = API_SUBMIT_RESULT.format(job_id=job_id)
            print(f"   🏢 [Production Mode] ส่งไปยัง Endpoint: {target_endpoint}", flush=True)

        submit_resp = _request("POST", target_endpoint, data=form_data)
        submit_resp.raise_for_status()

        engine_desc = {
            0: "Local AI ผ่านสมบูรณ์ (0)",
            120: "Gemini กู้สำเร็จ - ตัวเลขเบลอ (120)",
            1120: "Gemini กู้สำเร็จ - หากล่องไม่ครบ (1120)",
            110: "ส่งคนตรวจ - อ่านเลขไม่ออก (110)",
            1110: "ส่งคนตรวจ - หากล่องไม่พบ/ภาพมืด (1110)",
        }.get(ocr_engine, f"Score {ocr_engine}")

        conf_log = f" | confidence={form_data['confidence']}%" if "confidence" in form_data else ""
        sent_reading = form_data.get("ocr_reading", "-")
        print(f"✅ งาน #{job_id} ({'Test' if is_test else 'Production'}) เสร็จสมบูรณ์!", flush=True)
        print(f"   • ผลลัพธ์: error_type={error_type} | ocr_reading={sent_reading} | ocr_engine={ocr_engine} ({engine_desc}){conf_log}", flush=True)

    except Exception as exc:
        print(f"❌ เกิดข้อผิดพลาดกับงาน #{job_id}: {exc}", flush=True)
        try:
            _request("POST", API_SUBMIT_FAIL.format(job_id=job_id), json={"error": str(exc)[:2000]})
        except Exception:
            pass
    finally:
        # 🧹 ล้างไฟล์ภาพที่ดาวน์โหลดมาชั่วคราว เพื่อประหยัดพื้นที่ดิสก์ใน Container
        for p in image_paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# -------------------------------------------------------------------
# 🚀 Loop หลัก — ทำงานวนซ้ำตลอดเวลา
# -------------------------------------------------------------------
def run_forever() -> None:
    if not OCR_API_KEY:
        print("⚠️  กรุณาตั้งค่า OCR_API_KEY ในไฟล์ .env ก่อนเริ่มทำงาน (Container เปิดสแตนด์บายอยู่)", flush=True)
        while not OCR_API_KEY:
            time.sleep(30)

    print("=" * 65, flush=True)
    print(f"🚀 [ocr-client] เริ่มทำงาน เชื่อมต่อ: {IMAGE_STORE_BASE_URL}", flush=True)
    print(f"⏱️  ตรวจสอบคิวงานทุกๆ {POLL_INTERVAL_SECONDS} วินาที...", flush=True)
    print("=" * 65, flush=True)

    while True:
        try:
            jobs = fetch_queued_jobs()
            if jobs:
                print(f"\n📋 [Step 1] พบคิวงานใหม่ {len(jobs)} รายการ", flush=True)
                for job in jobs:
                    process_single_job(job)
            else:
                pass

        except Exception as exc:
            print(f"[ocr-client] ข้อผิดพลาดขณะวนลูป: {exc}", flush=True)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_forever()