"""
mock_server.py
--------------------------------------------------------------------------
Server จำลอง (Mock Server) สำหรับทดสอบ ocr_client_poller.py ในเครื่องตัวเอง
โดยไม่ต้องรอให้เพื่อนเปิด Server จริง

วิธีใช้งาน:
1. เปิด Terminal ที่ 1: python mock_server.py
2. เปิด Terminal ที่ 2: python ocr_client_poller.py
"""

import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

app = FastAPI(title="Mock Image-Store Server")

# รายการภาพทดสอบในเครื่องที่มีอยู่
SAMPLE_IMAGES = list(Path("elec").glob("*.jpg"))[:3]
if not SAMPLE_IMAGES:
    SAMPLE_IMAGES = list(Path(".").glob("*.jpg"))[:3]

# สร้างคิวงานจำลอง
MOCK_JOBS = [
    {
        "id": i + 1,
        "image_id": f"img_{i+1}",
        "filename": img.name,
        "status": "queued",
        "file_path": str(img)
    }
    for i, img in enumerate(SAMPLE_IMAGES)
]


@app.post("/login")
def login(data: dict):
    """จำลองการ Login"""
    print("🔑 [Mock Server] ได้รับการ Login จาก ocr-service")
    return {"access_token": "mock_token_12345"}


@app.get("/admin/images/ocr")
def get_ocr_jobs(job_status: str = "queued", limit: int = 5):
    """จำลองการส่งคิวงานที่ยังไม่ได้อ่านไปให้ OCR Client"""
    queued = [j for j in MOCK_JOBS if j["status"] == "queued"][:limit]
    if queued:
        print(f"📦 [Mock Server] ส่งงาน {len(queued)} รายการไปให้ OCR Client")
    return queued


@app.post("/admin/images/ocr/{job_id}/claim")
def claim_job(job_id: int):
    """จำลองการจองงาน"""
    for j in MOCK_JOBS:
        if j["id"] == job_id:
            j["status"] = "processing"
            print(f"🔒 [Mock Server] งาน #{job_id} ({j['filename']}) ถูกจองแล้ว")
            return {"status": "claimed"}
    raise HTTPException(status_code=404, detail="Job not found")


@app.get("/admin/images/{image_id}/file")
def get_image_file(image_id: str):
    """จำลองการดาวน์โหลดไฟล์ภาพ"""
    for j in MOCK_JOBS:
        if j["image_id"] == image_id:
            print(f"📤 [Mock Server] ส่งไฟล์ภาพ {j['filename']} ไปให้ OCR Client")
            return FileResponse(j["file_path"], media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Image not found")


@app.post("/admin/images/ocr/{job_id}/result")
async def receive_result(
    job_id: int,
    reading: float = Form(...),
    raw_text: str = Form(...),
    result_image: UploadFile = File(None)
):
    """จำลองการรับผลลัพธ์ที่ OCR อ่านเสร็จแล้วกลับมาบันทึก"""
    for j in MOCK_JOBS:
        if j["id"] == job_id:
            j["status"] = "completed"
            j["reading"] = reading
            j["raw_text"] = raw_text
            print("=" * 60)
            print(f"🎉 [Mock Server] ได้รับผลลัพธ์ของงาน #{job_id} ({j['filename']}) สำเร็จ!")
            print(f"   📊 ตัวเลขที่อ่านได้: {reading}")
            print(f"   📝 รายละเอียด: {raw_text}")
            print("=" * 60)
            return {"status": "success"}

    raise HTTPException(status_code=404, detail="Job not found")


@app.post("/admin/images/ocr/{job_id}/fail")
def job_failed(job_id: int, data: dict):
    print(f"❌ [Mock Server] งาน #{job_id} ล้มเหลว: {data.get('error')}")
    return {"status": "recorded"}


if __name__ == "__main__":
    print("🚀 Mock Server กำลังรันที่ http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
