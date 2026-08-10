import sys
import cv2
from ultralytics import YOLO

# 1. เช็คว่ามีการส่งชื่อไฟล์รูปภาพมาทาง Terminal หรือไม่
if len(sys.argv) < 2:
    print("❌ กรุณาใส่ชื่อไฟล์รูปภาพด้วย เช่น: python meter.py m1.JPG")
    sys.exit(1)

image_path = sys.argv[1]

# 2. อ่านภาพต้นฉบับ
img = cv2.imread(image_path)
if img is None:
    print(f"❌ ไม่สามารถเปิดไฟล์รูปภาพ {image_path} ได้ กรุณาเช็คชื่อไฟล์อีกครั้ง")
    sys.exit(1)

h, w, _ = img.shape

# 3. ครอปเน้นเฉพาะโซนหน้าปัดตัวเลข (ประมาณช่วง 25%-50% จากด้านบนของรูป)
crop_img = img[int(h*0.25):int(h*0.50), int(w*0.20):int(w*0.85)]

# 4. 🔥 [เพิ่มตรงนี้] ทำ CLAHE ขุดตัวเลขจากเงามืด และลบ Noise
gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray)
# แปลงกลับเป็น 3 channels (BGR) เพื่อส่งต่อให้ YOLO
processed_img = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

# เซฟภาพที่แต่งแล้วชั่วคราว
cv2.imwrite("temp_crop.jpg", processed_img)

# 5. โหลดโมเดล & Predict ภาพที่ผ่านการปรับแต่งแล้ว
model = YOLO("best.pt")

# ปรับ conf=0.28 เพื่อดีดฝุ่นที่มีค่าความมั่นใจต่ำออกไป และใส่ iou=0.4 กันกรอบซ้อน
results = model("temp_crop.jpg", imgsz=640, conf=0.28, iou=0.4, device="cpu")

detected_digits = []

# 6. ดึงข้อมูลตัวเลขและพิกัด X
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    label = model.names[cls_id]

    # สนใจเฉพาะ Class ตัวเลข 0-9
    if label.isdigit():
        x_min = float(box.xyxy[0][0])
        confidence = float(box.conf[0])
        detected_digits.append((x_min, label, confidence))

# 7. เรียงลำดับตัวเลขจากซ้ายไปขวาตามแกน X
detected_digits.sort(key=lambda item: item[0])
final_reading = "".join([item[1] for item in detected_digits])

print("\n==================================")
print(f"📷 ตัวเลขมิเตอร์ที่อ่านได้: {final_reading}")
print("==================================\n")

# แสดงภาพที่ YOLO ตีกรอบให้ดู
results[0].show()
