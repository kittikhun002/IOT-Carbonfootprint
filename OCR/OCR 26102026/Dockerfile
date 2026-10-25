FROM python:3.11-slim

# ติดตั้ง lib พื้นฐานสำหรับระบบประมวลผล
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ติดตั้ง Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดและไฟล์โมเดล AI
COPY . .

# ให้ Python แสดง log ออกหน้าจอทันที ไม่ค้างใน buffer
ENV PYTHONUNBUFFERED=1

# รัน poller เฝ้าคิวอัตโนมัติ
CMD ["python", "ocr_client_poller.py"]
