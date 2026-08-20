"""Main Pipeline: Smart Meter Reading System.

Workflow:
- detect_meter_type         : ตรวจจับประเภทมิเตอร์จากชื่อไฟล์ (e = elec, w = water, g = gas)
- step1_read_with_local_ai  : ขั้นที่ 1 ให้ YOLO + CNN อ่านภาพ
- step2_validate_rules      : ขั้นที่ 2 ตรวจสอบผลด้วย Rule-Base (2 กฎ)
- step3_verify_with_gemini  : ขั้นที่ 3 ส่งให้ Gemini Vision ช่วยอ่านซ้ำ
- step4_escalate_to_human   : ขั้นที่ 4 ส่งให้เจ้าหน้าที่ตรวจสอบ (Human Review)
- run_pipeline              : ประมวลผลภาพเดี่ยว 1 ภาพ
- run_multi_image_pipeline  : รวม 3 ภาพจากรอบเดียวกัน (ESP32) โหวตเหลือ 1 คำตอบที่ดีที่สุด
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

from meter_reader3 import read_meter
from meter_validator import validate_meter
from gemini_verifier import verify_with_gemini

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ==========================================================
# 🏷️ ตรวจจับประเภทมิเตอร์จากชื่อไฟล์ (Prefix Detection)
# ==========================================================
def detect_meter_type(filename_or_path: str) -> str:
    """
    ตรวจจับประเภทมิเตอร์จากตัวอักษรขึ้นต้นของชื่อไฟล์:
    - e100_... -> elec (มิเตอร์ไฟฟ้า)
    - w100_... -> water (มิเตอร์น้ำ)
    - g100_... -> gas (มิเตอร์แก๊ส)
    """
    name = Path(filename_or_path).name.lower()
    if name.startswith("e"):
        return "elec"
    elif name.startswith("w"):
        return "water"
    elif name.startswith("g"):
        return "gas"
    return "auto"


# ==========================================================
# 🔹 ขั้นตอนที่ 1: อ่านภาพด้วย Local AI (YOLO + CNN)
# ==========================================================
def step1_read_with_local_ai(image_path: str, meter_type: str = "auto", expected_digits: int = None):
    print(f"\n[ขั้นตอนที่ 1] 👁️ กำลังอ่านภาพด้วย Local AI (YOLO + CNN)... ({Path(image_path).name})")
    local_data = read_meter(image_path, expected_digits=expected_digits, meter_type=meter_type)
    return local_data


# ==========================================================
# 🔹 ขั้นตอนที่ 2: ตรวจสอบด้วย Rule-Based
# ==========================================================
def step2_validate_rules(results: list, min_conf: float = 0.60):
    print(f"[ขั้นตอนที่ 2] ⚖️ กำลังตรวจสอบความถูกต้องด้วย Rule-Base...")
    is_valid, errors = validate_meter(results, min_conf=min_conf)

    if is_valid:
        print("  • ผลการตรวจ: ✅ ผ่านเกณฑ์ทุกข้อ (ตัวเลขชัดเจน & กลไกเฟืองถูกต้อง)")
    else:
        print("  • ผลการตรวจ: ❌ ไม่ผ่านเกณฑ์ พบปัญหา:")
        for err in errors:
            print(f"    - {err}")

    return is_valid, errors


# ==========================================================
# 🔹 ขั้นตอนที่ 3: ส่งต่อให้ Gemini Vision ช่วยวิเคราะห์
# ==========================================================
def step3_verify_with_gemini(image_path: str, errors: list, gemini_key: str = None):
    print(f"\n[ขั้นตอนที่ 3] 🤖 ส่งภาพให้ Gemini Vision ช่วยตรวจสอบ...")
    error_summary = "; ".join(errors)
    gemini_res = verify_with_gemini(image_path, error_notes=error_summary, api_key=gemini_key)

    if gemini_res.get("success"):
        print(f"  • Gemini ตอบกลับ: ✅ อ่านได้ค่า '{gemini_res['reading']}'")
        print(f"    เหตุผล: {gemini_res.get('reason')}")
    else:
        print(f"  • Gemini ตอบกลับ: ❌ ไม่สามารถยืนยันได้ ({gemini_res.get('error')})")

    return gemini_res


# ==========================================================
# 🔹 ขั้นตอนที่ 4: ส่งต่อให้เจ้าหน้าที่ตรวจสอบ (Human Review)
# ==========================================================
def step4_escalate_to_human(image_path: str, local_errors: list, gemini_error: str = None):
    print(f"\n[ขั้นตอนที่ 4] 🚩 ส่งต่อให้เจ้าหน้าที่ตรวจสอบ (Human Review Required)")
    return {
        "status": "HUMAN_REVIEW_REQUIRED",
        "image_path": str(image_path),
        "local_errors": local_errors,
        "gemini_error": gemini_error
    }


# ==========================================================
# 🚀 ประมวลผลภาพเดี่ยว (Single Image)
# ==========================================================
def run_pipeline(image_path: str, meter_type: str = "auto", expected_digits: int = None, min_conf: float = 0.60, gemini_key: str = None):
    # ออโต้ดีเทคประเภทมิเตอร์ถ้าไม่ได้ระบุมา
    if meter_type == "auto":
        meter_type = detect_meter_type(image_path)

    local_data = step1_read_with_local_ai(image_path, meter_type, expected_digits)
    if not local_data:
        return step4_escalate_to_human(image_path, local_errors=["ไม่สามารถเปิดหรือประมวลผลภาพได้"])

    results = local_data["results"]
    formatted = local_data["formatted"]

    is_valid, errors = step2_validate_rules(results, min_conf=min_conf)

    if is_valid:
        return {
            "status": "APPROVED_LOCAL",
            "reading": formatted["fmt_a"],
            "raw": formatted["raw_a"],
            "image_path": str(image_path),
            "meter_type": meter_type,
            "confidence": sum(r["confidence"] for r in results) / len(results) if results else 0.0
        }

    gemini_res = step3_verify_with_gemini(image_path, errors, gemini_key)
    if gemini_res.get("success"):
        return {
            "status": "APPROVED_GEMINI",
            "reading": gemini_res["reading"],
            "raw": "".join(c for c in gemini_res["reading"] if c.isdigit()),
            "reason": gemini_res.get("reason"),
            "image_path": str(image_path),
            "meter_type": meter_type
        }

    return step4_escalate_to_human(image_path, local_errors=errors, gemini_error=gemini_res.get("error"))


# ==========================================================
# 🗳️ ประมวลผลชุดภาพ (ESP32 ถ่าย 3 ภาพ ➔ 1 คำตอบ)
# ==========================================================
def run_multi_image_pipeline(image_paths: list, meter_type: str = "auto", expected_digits: int = None, min_conf: float = 0.60, gemini_key: str = None):
    """
    รับลิสต์ภาพ 3 ภาพ (เช่น จาก ESP32) -> อ่านทุกภาพ -> โหวตเสียงส่วนใหญ่เหลือ 1 คำตอบ
    """
    print("=" * 65)
    print(f"📸 เริ่มประมวลผลชุดภาพ ESP32 ({len(image_paths)} ภาพ)")
    print("=" * 65)

    if not image_paths:
        return {"status": "HUMAN_REVIEW_REQUIRED", "error": "ไม่มีไฟล์ภาพ"}

    # ถ้าไม่ได้ระบุประเภท ให้ตรวจจากภาพแรก
    if meter_type == "auto":
        meter_type = detect_meter_type(image_paths[0])
    print(f"🏷️  ประเภทมิเตอร์ที่ตรวจพบอัตโนมัติ: {meter_type.upper()}")

    all_results = []
    for img_p in image_paths:
        res = run_pipeline(img_p, meter_type=meter_type, expected_digits=expected_digits, min_conf=min_conf, gemini_key=gemini_key)
        all_results.append(res)

    # 1. กรองเฉพาะภาพที่ Local AI อ่านผ่าน
    approved_locals = [r for r in all_results if r.get("status") == "APPROVED_LOCAL"]

    if approved_locals:
        raw_votes = [r["raw"] for r in approved_locals]
        vote_counts = Counter(raw_votes)
        best_raw, count = vote_counts.most_common(1)[0]

        # กรณีเสียงส่วนใหญ่ตรงกัน (เช่น 2 ใน 3 รูป หรือ 3 ใน 3)
        if count >= 2 or len(approved_locals) == 1:
            best_item = next(r for r in approved_locals if r["raw"] == best_raw)
            print(f"\n🎉 [มติเอกฉันท์ Majority Vote {count}/{len(image_paths)}] ได้ผลลัพธ์: {best_item['reading']}")
            return {
                "status": "APPROVED_LOCAL",
                "reading": best_item["reading"],
                "raw": best_item["raw"],
                "meter_type": meter_type,
                "vote_ratio": f"{count}/{len(image_paths)}",
                "selected_image": best_item["image_path"]
            }

        # ถ้าคะแนนเสียงเท่ากัน ให้เลือกภาพที่ Confidence รวมสูงสุด
        best_conf_item = max(approved_locals, key=lambda x: x.get("confidence", 0.0))
        print(f"\n🎉 [เลือกภาพที่ชัดที่สุด Highest Confidence] ได้ผลลัพธ์: {best_conf_item['reading']}")
        return best_conf_item

    # 2. ถ้า Local AI ไม่ผ่าน ให้เช็คผลจาก Gemini
    gemini_approved = [r for r in all_results if r.get("status") == "APPROVED_GEMINI"]
    if gemini_approved:
        print(f"\n✨ [ยืนยันโดย Gemini Vision] ได้ผลลัพธ์: {gemini_approved[0]['reading']}")
        return gemini_approved[0]

    # 3. ถ้าไม่มีภาพไหนผ่านเลย -> ส่งคนตรวจ
    print("\n🚩 [ทุกภาพไม่ผ่านเกณฑ์ -> ส่งต่อให้คนตรวจ]")
    return {
        "status": "HUMAN_REVIEW_REQUIRED",
        "meter_type": meter_type,
        "all_results": all_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Meter Multi-Image Pipeline")
    parser.add_argument("images", nargs="+", help="Paths to meter images (1 or more)")
    parser.add_argument("--type", choices=["auto", "elec", "gas", "water"], default="auto")
    parser.add_argument("--digits", type=int, default=None)
    parser.add_argument("--min-conf", type=float, default=0.60)
    parser.add_argument("--gemini-key", type=str, default=None)

    args = parser.parse_args()

    if len(args.images) == 1:
        run_pipeline(args.images[0], meter_type=args.type, expected_digits=args.digits, min_conf=args.min_conf, gemini_key=args.gemini_key)
    else:
        run_multi_image_pipeline(args.images, meter_type=args.type, expected_digits=args.digits, min_conf=args.min_conf, gemini_key=args.gemini_key)
