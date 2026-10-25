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
    if name.startswith(("ele", "e")):
        return "elec"
    elif name.startswith(("wat", "w")):
        return "water"
    elif name.startswith(("gas", "g")):
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
def step2_validate_rules(results: list, min_conf: float = 0.60, current_reading: float = None, history: list = None, meter_type: str = "auto"):
    print(f"[ขั้นตอนที่ 2] ⚖️ กำลังตรวจสอบความถูกต้องด้วย Rule-Base...")
    is_valid, errors = validate_meter(results, min_conf=min_conf, current_reading=current_reading, history=history, meter_type=meter_type)

    if is_valid:
        print("  • ผลการตรวจ: ✅ ผ่านเกณฑ์ทุกข้อ (ตัวเลขชัดเจน & กลไกเฟืองถูกต้อง & ประวัติสมเหตุสมผล)")
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
def step4_escalate_to_human(image_path: str, local_errors: list, gemini_error: str = None, ocr_engine: int = 110, reading: str = None, raw: str = None):
    print(f"\n[ขั้นตอนที่ 4] 🚩 ส่งต่อให้เจ้าหน้าที่ตรวจสอบ (Human Review Required)")
    return {
        "status": "HUMAN_REVIEW_REQUIRED",
        "image_path": str(image_path),
        "reading": reading,
        "raw": raw,
        "local_errors": local_errors,
        "gemini_error": gemini_error,
        "ocr_engine": ocr_engine
    }


# ==========================================================
# 🚀 ประมวลผลภาพเดี่ยว (Single Image)
# ==========================================================
def run_pipeline(image_path: str, meter_type: str = "auto", expected_digits: int = None, min_conf: float = 0.60, history: list = None, gemini_key: str = None):
    # ออโต้ดีเทคประเภทมิเตอร์ถ้าไม่ได้ระบุมา
    if meter_type == "auto":
        meter_type = detect_meter_type(image_path)

    score = 0  # แต้มสะสม Summation Score (YOLO: พัน, CNN: ร้อย, Gemini: สิบ)

    # 🌟 ถ้าไม่มีประวัติเดิม (ค่าแรก) -> ให้ Gemini อ่านและส่งผลผ่านทันที
    if not history:
        gemini_res = step3_verify_with_gemini(image_path, errors=["ไม่มีประวัติเดิม"], gemini_key=gemini_key)
        if gemini_res.get("success"):
            return {
                "status": "APPROVED_GEMINI",
                "reading": gemini_res["reading"],
                "raw": "".join(c for c in str(gemini_res["reading"]) if c.isdigit()),
                "meter_type": meter_type,
                "confidence": 0.98,
                "ocr_engine": 120
            }

    # 🔹 Stage 1: อ่านภาพด้วย Local AI (YOLO + CNN)
    local_data = step1_read_with_local_ai(image_path, meter_type, expected_digits)
    if not local_data:
        score += 1000 + 100 + 10  # เปิดภาพไม่ได้ -> 1110
        return step4_escalate_to_human(image_path, local_errors=["ไม่สามารถเปิดหรือประมวลผลภาพได้"], ocr_engine=score)

    results = local_data["results"]
    formatted = local_data["formatted"]

    raw_str = str(formatted.get("raw_a", ""))
    val_str = formatted.get("fmt_a", "").split()[0] if formatted.get("fmt_a") else raw_str
    raw_val = float(val_str) if val_str.replace(".", "", 1).isdigit() else None

    # ตรวจสอบว่า YOLO เจอกล่องครบตามประเภทมิเตอร์หรือไม่ (หลักพัน)
    valid_counts = {"elec": {5}, "gas": {8}, "water": {4, 5, 7, 8}}
    yolo_ok = len(results) in valid_counts.get(meter_type, {len(results)})
    if not yolo_ok:
        score += 1000  # YOLO หากล่องไม่ครบ บวก 1000

    # 🔹 Stage 2: ตรวจสอบความถูกต้องด้วย Rule-Base (หลักร้อย)
    is_valid, errors = step2_validate_rules(results, min_conf=min_conf, current_reading=raw_val, history=history, meter_type=meter_type)
    if not is_valid:
        score += 100  # ตัวเลขอ่านไม่ออก / เบลอ / Rule ไม่ผ่าน บวก 100

    # กรณี Local AI ผ่านฉลุยทั้ง 2 ด่าน (score ยังคงเป็น 0)
    if score == 0:
        return {
            "status": "APPROVED_LOCAL",
            "reading": formatted["fmt_a"],
            "raw": formatted["raw_a"],
            "image_path": str(image_path),
            "meter_type": meter_type,
            "confidence": sum(r["confidence"] for r in results) / len(results) if results else 0.0,
            "ocr_engine": score  # 0
        }

    # 🔹 Stage 3: ส่งให้ Gemini Vision ช่วยกู้ผลลัพธ์ (หลักสิบ)
    gemini_res = step3_verify_with_gemini(image_path, errors, gemini_key)
    if gemini_res.get("success"):
        gemini_reading = str(gemini_res.get("reading", ""))
        raw_digits = "".join(c for c in gemini_reading if c.isdigit())
        num_str = "".join(c for c in gemini_reading if c.isdigit() or c == ".")
        gemini_val = float(num_str) if num_str.replace(".", "", 1).isdigit() else None

        gemini_results = [{"pos": i + 1, "val_after": ch, "confidence": 1.0} for i, ch in enumerate(raw_digits)]

        print(f"\n[ขั้นตอนที่ 3.1] ⚖️ ตรวจสอบผลลัพธ์ของ Gemini ด้วย Rule-Base...")
        g_valid, g_errors = step2_validate_rules(
            gemini_results, min_conf=min_conf, current_reading=gemini_val, history=history, meter_type=meter_type
        )

        if g_valid:
            score += 20  # Gemini กู้สำเร็จและผ่านทุกกฎ บวก 20 (รวมเป็น 120 หรือ 1120)
            return {
                "status": "APPROVED_GEMINI",
                "reading": gemini_res["reading"],
                "raw": raw_digits,
                "reason": gemini_res.get("reason"),
                "image_path": str(image_path),
                "meter_type": meter_type,
                "confidence": 0.95,
                "ocr_engine": score
            }
        else:
            # Gemini ตอบมาแต่ไม่ผ่านกฎ (เช่น จำนวนหลักผิด หรือค่าลดลง) -> ส่งคนตรวจ
            print(f"  • ผลตรวจ Gemini: ❌ ไม่ผ่านเกณฑ์ Rule-Base -> ส่งคนตรวจ")
            score += 10  # Gemini กู้ไม่สำเร็จ บวก 10 (รวมเป็น 110 หรือ 1110)
            return step4_escalate_to_human(
                image_path,
                local_errors=errors,
                gemini_error="; ".join(g_errors),
                ocr_engine=score,
                reading=gemini_res.get("reading") or formatted.get("fmt_a"),
                raw=raw_digits or formatted.get("raw_a")
            )
    else:
        score += 10  # Gemini กู้ไม่สำเร็จ บวก 10 (รวมเป็น 110 หรือ 1110)
        return step4_escalate_to_human(
            image_path,
            local_errors=errors,
            gemini_error=gemini_res.get("error"),
            ocr_engine=score,
            reading=formatted.get("fmt_a"),
            raw=formatted.get("raw_a")
        )


# ==========================================================
# 🗳️ ประมวลผลชุดภาพ (ESP32 ถ่าย 3 ภาพ ➔ 1 คำตอบ)
# ==========================================================
def run_multi_image_pipeline(image_paths: list, meter_type: str = "auto", expected_digits: int = None, min_conf: float = 0.60, history: list = None, gemini_key: str = None):
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
        res = run_pipeline(img_p, meter_type=meter_type, expected_digits=expected_digits, min_conf=min_conf, history=history, gemini_key=gemini_key)
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
                "selected_image": best_item["image_path"],
                "confidence": best_item.get("confidence", 0.0),
                "ocr_engine": 0
            }

        # ถ้าคะแนนเสียงเท่ากัน ให้เลือกภาพที่ Confidence รวมสูงสุด
        best_conf_item = max(approved_locals, key=lambda x: x.get("confidence", 0.0))
        best_conf_item["ocr_engine"] = 0
        best_conf_item.setdefault("selected_image", best_conf_item.get("image_path"))
        best_conf_item.setdefault("vote_ratio", f"1/{len(image_paths)}")
        print(f"\n🎉 [เลือกภาพที่ชัดที่สุด Highest Confidence] ได้ผลลัพธ์: {best_conf_item['reading']}")
        return best_conf_item

    # 2. ถ้า Local AI ไม่ผ่าน ให้เช็คผลจาก Gemini ด้วย Majority Vote 2 ใน 3
    gemini_approved = [r for r in all_results if r.get("status") == "APPROVED_GEMINI"]
    if gemini_approved:
        g_raw_votes = [r["raw"] for r in gemini_approved]
        g_vote_counts = Counter(g_raw_votes)
        g_best_raw, g_count = g_vote_counts.most_common(1)[0]

        # ต้องมีเสียงโหวตตรงกันอย่างน้อย 2 ใน 3 ภาพ (หรือถ้ามีภาพเดียวให้อนุมัติตามภาพนั้น)
        if g_count >= 2 or len(gemini_approved) == 1:
            g_best_item = next(r for r in gemini_approved if r["raw"] == g_best_raw)
            print(f"\n✨ [ยืนยันโดย Gemini Vision Majority Vote {g_count}/{len(image_paths)}] ได้ผลลัพธ์: {g_best_item['reading']}")
            return {
                "status": "APPROVED_GEMINI",
                "reading": g_best_item["reading"],
                "raw": g_best_item["raw"],
                "reason": g_best_item.get("reason"),
                "meter_type": meter_type,
                "vote_ratio": f"{g_count}/{len(image_paths)}",
                "selected_image": g_best_item["image_path"],
                "confidence": g_best_item.get("confidence", 0.95),
                "ocr_engine": g_best_item.get("ocr_engine", 120)
            }
        else:
            print(f"\n⚠️ [Gemini Vision อ่านได้ผลไม่ตรงกัน ({g_count}/{len(image_paths)}) -> ส่งต่อให้คนตรวจ]")

    # 3. ถ้าไม่มีภาพไหนผ่านเลย -> ส่งคนตรวจ
    all_errs = []
    for r in all_results:
        all_errs.extend(r.get("local_errors", []))
        if r.get("gemini_error"):
            all_errs.append(r["gemini_error"])
    if not all_errs:
        all_errs = ["ภาพอ่านไม่ออกหรือไม่พบช่องตัวเลข"]

    overall_engine = 1110 if (all_results and all(r.get("ocr_engine") == 1110 for r in all_results)) else 110
    # แนบตัวเลขที่ AI อ่านได้ไปด้วย เพื่อไม่ให้กลายเป็น 0
    best_candidate = next((r for r in all_results if r.get("raw")), {})

    print("\n🚩 [ทุกภาพไม่ผ่านเกณฑ์ -> ส่งต่อให้คนตรวจ]")
    return {
        "status": "HUMAN_REVIEW_REQUIRED",
        "meter_type": meter_type,
        "reading": best_candidate.get("reading", ""),
        "raw": best_candidate.get("raw", ""),
        "local_errors": all_errs,
        "all_results": all_results,
        "ocr_engine": overall_engine
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
