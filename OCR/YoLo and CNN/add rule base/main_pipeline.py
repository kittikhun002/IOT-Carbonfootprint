"""Main Pipeline: End-to-End Smart Meter Reading System.

Workflow:
1. Local AI: YOLO (Wheel Localization) + CNN (Digit & Rolling Detection)
2. Rule-Based Engine:
   - Check 1: Confidence & Missing digit check
   - Check 2: Mechanical Gear Consistency rule
3. Secondary AI (Gemini Vision Fallback):
   - Invoked only if Local AI fails Rule-Based validation
4. Human-In-The-Loop Escalation:
   - Triggered if both Local AI and Gemini fail validation
"""

import sys
import argparse
from pathlib import Path
from meter_reader3 import read_meter
from meter_validator import MeterValidator
from gemini_verifier import GeminiVerifier

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def process_meter_image(
    image_path: str,
    meter_type: str = "auto",
    expected_digits: int = None,
    previous_reading: float = None,
    min_confidence: float = 0.60,
    gemini_api_key: str = None,
):
    print("=" * 60)
    print(f"🚀 เริ่มกระบวนการประมวลผลมิเตอร์: {image_path}")
    print("=" * 60)

    # ----------------------------------------------------
    # Stage 1: Local AI Inference (YOLO + CNN)
    # ----------------------------------------------------
    local_output = read_meter(image_path, expected_digits=expected_digits, meter_type=meter_type)
    if not local_output or not local_output.get("results"):
        print("❌ ไม่สามารถประมวลผลภาพด้วย Local AI ได้")
        return {"status": "HUMAN_REVIEW_REQUIRED", "reason": "Local inference failed entirely"}

    results = local_output["results"]
    formatted = local_output["formatted"]

    # ----------------------------------------------------
    # Stage 2: Rule-Based Validation
    # ----------------------------------------------------
    validator = MeterValidator(min_digit_confidence=min_confidence)
    report = validator.validate_reading_results(results, previous_reading=previous_reading)

    print("\n" + report.summary())

    # Case 1: ผ่านทุกกฎเกณฑ์
    if report.is_valid:
        print("\n✨ [STATUS: APPROVED_LOCAL]")
        print(f"🎉 ผ่านการตรวจสอบความถูกต้อง 100%! ผลลัพธ์: {formatted['fmt_a']}")
        return {
            "status": "APPROVED_LOCAL",
            "reading": formatted["fmt_a"],
            "raw": formatted["raw_a"],
            "validation_report": report,
        }

    # Case 2: ไม่ผ่านกฎ -> ส่งให้ Gemini วิเคราะห์ต่อ
    print("\n⚠️  [STAGE: ESCALATING TO GEMINI VISION]")
    notes = "; ".join(report.reasons)
    verifier = GeminiVerifier(api_key=gemini_api_key)

    if not verifier.is_available():
        print("⚠️  ยังไม่ได้ตั้งค่า GEMINI_API_KEY (ข้ามไปยัง Human Review)")
        print("\n🚩 [STATUS: HUMAN_REVIEW_REQUIRED]")
        print("ส่งต่อไปยังเจ้าหน้าที่ตรวจสอบเนื่องจากไม่ผ่าน Rule-base และไม่มี Gemini Fallback")
        return {
            "status": "HUMAN_REVIEW_REQUIRED",
            "reasons": report.reasons,
            "local_reading": formatted["fmt_a"],
        }

    print("🤖 กำลังส่งภาพให้ Gemini Vision วิเคราะห์และตรวจสอบความถูกต้อง...")
    gemini_res = verifier.verify_image(image_path, context_notes=notes)

    if gemini_res.get("success"):
        g_data = gemini_res["data"]
        print(f"✅ Gemini ตอบกลับ: อ่านค่าได้ = {g_data.get('meter_reading_raw')}")
        print(f"   เหตุผลจาก Gemini: {g_data.get('reasoning')}")

        # ตรวจสอบผลจาก Gemini อีกรอบ
        g_digits = [
            {
                "pos": d.get("pos", idx + 1),
                "val_after": str(d.get("digit", "")),
                "confidence": float(d.get("confidence", 0.90)),
                "is_transition": d.get("is_transition", False),
                "display": str(d.get("digit", "")),
            }
            for idx, d in enumerate(g_data.get("digits", []))
        ]
        g_report = validator.validate_reading_results(g_digits, previous_reading=previous_reading)

        if g_report.is_valid:
            print("\n✨ [STATUS: APPROVED_GEMINI]")
            print(f"🎉 ผ่านการตรวจสอบโดย Gemini Fallback! ผลลัพธ์: {g_data.get('meter_reading_raw')}")
            return {
                "status": "APPROVED_GEMINI",
                "reading": g_data.get("meter_reading_raw"),
                "gemini_data": g_data,
            }

    # Case 3: Gemini ยังไม่ผ่าน หรือเรียกไม่สำเร็จ -> ส่งให้คนตรวจ
    print("\n🚩 [STATUS: HUMAN_REVIEW_REQUIRED]")
    print("ส่งต่อไปยังคิวเจ้าหน้าที่ตรวจสอบ (Human-in-the-loop) เนื่องจากทั้ง Local AI และ Gemini ไม่สามารถยืนยันความถูกต้องได้")
    return {
        "status": "HUMAN_REVIEW_REQUIRED",
        "local_reasons": report.reasons,
        "gemini_response": gemini_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Smart Meter Reading Pipeline")
    parser.add_argument("image_path", help="Path to meter image")
    parser.add_argument("--type", choices=["auto", "elec", "gas", "water"], default="auto", help="Meter type")
    parser.add_argument("--digits", type=int, default=None, help="Expected number of digits")
    parser.add_argument("--prev", type=float, default=None, help="Previous reading value to check non-decreasing")
    parser.add_argument("--min-conf", type=float, default=0.60, help="Minimum confidence threshold")
    parser.add_argument("--gemini-key", type=str, default=None, help="Gemini API Key")

    args = parser.parse_args()
    process_meter_image(
        image_path=args.image_path,
        meter_type=args.type,
        expected_digits=args.digits,
        previous_reading=args.prev,
        min_confidence=args.min_conf,
        gemini_api_key=args.gemini_key,
    )
