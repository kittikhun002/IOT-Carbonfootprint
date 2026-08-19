"""Meter Validator: Rule-Based Validation Engine for Industrial / Utility Meter Readings.

Rules implemented:
1. Confidence & Valid Character Check (ความมั่นใจของแต่ละหลัก)
2. Mechanical Gear Consistency Rule (กฎความสัมพันธ์ของฟันเฟืองมิเตอร์)
3. (Optional) Monotonic Increasing Check (กฎค่าน้ำ/ไฟต้องไม่ลด)
4. (Optional) Consumption Anomaly Check (ตรวจสอบอัตราการใช้งานผิดปกติ)
"""

import sys
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class ValidationReport:
    is_valid: bool
    confidence_passed: bool
    gear_consistency_passed: bool
    historical_passed: bool = True
    reasons: List[str] = field(default_factory=list)
    wheel_details: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.is_valid else "❌ FAIL"
        lines = [f"--- ผลการตรวจสอบ Rule-Based: {status} ---"]
        if self.reasons:
            lines.append("สาเหตุที่ต้องตรวจสอบเพิ่มเติม:")
            for r in self.reasons:
                lines.append(f"  • {r}")
        else:
            lines.append("  • ข้อมูลผ่านเกณฑ์ความเชื่อมั่นและกลไกฟันเฟืองสมบูรณ์")
        return "\n".join(lines)


class MeterValidator:
    """Validator that checks meter reading results against physical and statistical rules."""

    def __init__(
        self,
        min_digit_confidence: float = 0.60,
        allow_extrapolated_unknown: bool = False,
    ):
        self.min_digit_confidence = min_digit_confidence
        self.allow_extrapolated_unknown = allow_extrapolated_unknown

    def validate_reading_results(
        self,
        results: List[Dict[str, Any]],
        previous_reading: Optional[float] = None,
        max_daily_usage: Optional[float] = None,
        days_elapsed: float = 30.0,
    ) -> ValidationReport:
        """Run full validation pipeline on the extracted wheel results."""
        reasons = []
        conf_passed = True
        gear_passed = True
        hist_passed = True

        # ----------------------------------------------------
        # Rule 1: Confidence & Unknown Digit Check
        # ----------------------------------------------------
        for r in results:
            pos = r.get("pos", "?")
            val_after = r.get("val_after", "?")
            conf = float(r.get("confidence", 0.0))

            # 1.1 เช็คตัวเลขที่เป็นเครื่องหมายคำถามหรือไม่สามารถระบุได้
            if val_after == "?" or not val_after.isdigit():
                conf_passed = False
                reasons.append(f"[Rule 1 - Unclear Digit] ช่องที่ #{pos} อ่านค่าตัวเลขไม่ชัดเจน (ได้ค่า '{val_after}')")
            # 1.2 เช็คค่าความมั่นใจ (Confidence Threshold)
            elif conf < self.min_digit_confidence:
                conf_passed = False
                reasons.append(
                    f"[Rule 1 - Low Confidence] ช่องที่ #{pos} ค่าความมั่นใจต่ำเกินไป ({conf:.2f} < {self.min_digit_confidence:.2f})"
                )

        # ----------------------------------------------------
        # Rule 2: Mechanical Gear Consistency Check (กฎฟันเฟือง)
        # ----------------------------------------------------
        # ในระบบเฟืองมิเตอร์ (Odometer/Mechanical Register):
        # หลักทางซ้าย (หลักที่ i) จะเริ่มหมุน/คาบเกี่ยว (Transition) ได้ก็ต่อเมื่อ
        # หลักทางขวาที่อยู่ติดกัน (หลักที่ i+1) กำลังอยู่ในช่วงรอยต่อรอบการหมุน (เลข 8, 9, 0 หรือกำลัง Transition)
        n = len(results)
        for i in range(n - 1):
            curr_wheel = results[i]      # หลักซ้าย (เช่น หลักสิบ)
            right_wheel = results[i + 1] # หลักขวา (เช่น หลักหน่วย)

            pos_curr = curr_wheel.get("pos", i + 1)
            pos_right = right_wheel.get("pos", i + 2)

            is_curr_trans = curr_wheel.get("is_transition", False)
            is_right_trans = right_wheel.get("is_transition", False)

            val_right = right_wheel.get("val_after", "")

            if is_curr_trans:
                # ถ้าหลักซ้ายกำลังเปลี่ยนเลข แต่หลักขวาไม่ใช่ช่วงปลายรอบ (เช่น อยู่ที่เลข 1-7 นิ่งๆ)
                if val_right.isdigit() and int(val_right) in [1, 2, 3, 4, 5, 6, 7] and not is_right_trans:
                    gear_passed = False
                    reasons.append(
                        f"[Rule 2 - Gear Conflict] ช่อง #{pos_curr} ถูกตรวจพบว่ากำลังหมุน {curr_wheel.get('display')} "
                        f"แต่ช่องขวา #{pos_right} เพิ่งอยู่ที่เลข '{val_right}' (ผิดธรรมชาติของฟันเฟืองมิเตอร์)"
                    )

        # ----------------------------------------------------
        # Rule 3 & 4: Historical & Consumption (ถ้ามีการส่งค่ามาเทียบ)
        # ----------------------------------------------------
        if previous_reading is not None:
            raw_val_str = "".join(r.get("val_after", "0") for r in results)
            if raw_val_str.isdigit():
                current_val = float(raw_val_str)
                if current_val < previous_reading:
                    hist_passed = False
                    reasons.append(
                        f"[Rule 3 - Non-Decreasing] ค่าปัจจุบัน ({current_val}) น้อยกว่าค่ารอบก่อนหน้า ({previous_reading})"
                    )
                elif max_daily_usage is not None:
                    usage = current_val - previous_reading
                    max_allowed = max_daily_usage * days_elapsed
                    if usage > max_allowed:
                        hist_passed = False
                        reasons.append(
                            f"[Rule 4 - Usage Anomaly] ปริมาณการใช้ ({usage:.2f}) สูงผิดปกติเกินขีดจำกัด ({max_allowed:.2f})"
                        )

        is_all_valid = conf_passed and gear_passed and hist_passed
        return ValidationReport(
            is_valid=is_all_valid,
            confidence_passed=conf_passed,
            gear_consistency_passed=gear_passed,
            historical_passed=hist_passed,
            reasons=reasons,
            wheel_details=results,
        )


if __name__ == "__main__":
    # Test cases to verify the rules
    validator = MeterValidator(min_digit_confidence=0.60)

    # Case A: Normal Reading (Pass)
    test_results_pass = [
        {"pos": 1, "val_after": "0", "confidence": 0.95, "is_transition": False, "display": "0"},
        {"pos": 2, "val_after": "4", "confidence": 0.88, "is_transition": False, "display": "4"},
        {"pos": 3, "val_after": "2", "confidence": 0.91, "is_transition": False, "display": "2"},
        {"pos": 4, "val_after": "8", "confidence": 0.85, "is_transition": False, "display": "8"},
    ]
    report_a = validator.validate_reading_results(test_results_pass)
    print("Test A (Normal):", report_a.summary())

    # Case B: Gear Conflict (Wheel 2 is transitioning, but Wheel 3 is stable at digit 4)
    test_results_gear_fail = [
        {"pos": 1, "val_after": "0", "confidence": 0.95, "is_transition": False, "display": "0"},
        {"pos": 2, "val_after": "5", "confidence": 0.75, "is_transition": True, "display": "[4->5]"},
        {"pos": 3, "val_after": "4", "confidence": 0.90, "is_transition": False, "display": "4"},  # Gear conflict!
        {"pos": 4, "val_after": "1", "confidence": 0.85, "is_transition": False, "display": "1"},
    ]
    report_b = validator.validate_reading_results(test_results_gear_fail)
    print("\nTest B (Gear Conflict):", report_b.summary())
