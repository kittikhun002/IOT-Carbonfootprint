"""Meter Validator: Rule-Based Validation Engine for Industrial / Utility Meter Readings.

Rules implemented:
1. check_confidence: ตรวจสอบความมั่นใจของตัวเลขแต่ละหลัก (Confidence Threshold)
2. check_gear_consistency: ตรวจสอบความสัมพันธ์ของฟันเฟืองมิเตอร์ (Mechanical Gear Consistency)
3. validate_meter: ฟังก์ชันหลักที่รวมการตรวจทั้ง 2 กฎเข้าด้วยกัน
"""

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def check_confidence(results, min_conf=0.60):
    """
    กฎข้อที่ 1: ตรวจสอบความมั่นใจ (Confidence)
    - ต้องไม่มีตัว '?'
    - ค่าความมั่นใจต้องไม่ต่ำกว่า min_conf
    """
    errors = []
    for r in results:
        pos = r.get("pos", "?")
        val = r.get("val_after", "?")
        conf = float(r.get("confidence", 0.0))

        if val == "?" or not str(val).isdigit():
            errors.append(f"[Rule 1] ช่องที่ {pos} อ่านค่าตัวเลขไม่ชัดเจน (ได้ค่า '{val}')")
        elif conf < min_conf:
            errors.append(f"[Rule 1] ช่องที่ {pos} ความมั่นใจต่ำ ({conf:.2f} < {min_conf:.2f})")

    return errors


def check_gear_consistency(results):
    """
    กฎข้อที่ 2: ตรวจสอบกลไกฟันเฟือง (Mechanical Gear Consistency)
    - ถ้าหลักซ้ายกำลังหมุน (is_transition) หลักขวาต้องเป็นเลข 8, 9, 0 หรือกำลังหมุนด้วย
    - ถ้าหลักซ้ายหมุน แต่หลักขวาเป็นเลข 1, 2, 3, 4, 5, 6, 7 นิ่งๆ -> ผิดธรรมชาติฟันเฟือง
    """
    errors = []
    n = len(results)

    for i in range(n - 1):
        curr_wheel = results[i]       # หลักซ้าย (เช่น หลักสิบ)
        right_wheel = results[i + 1]  # หลักขวา (เช่น หลักหน่วย)

        is_curr_trans = curr_wheel.get("is_transition", False)
        is_right_trans = right_wheel.get("is_transition", False)
        val_right = str(right_wheel.get("val_after", ""))

        # ถ้าหลักซ้ายกำลังหมุน
        if is_curr_trans:
            # แต่หลักขวาเพิ่งอยู่ที่เลข 1 ถึง 7 นิ่งๆ
            if val_right.isdigit() and int(val_right) in [1, 2, 3, 4, 5, 6, 7] and not is_right_trans:
                errors.append(
                    f"[Rule 2] ช่อง #{curr_wheel.get('pos', i+1)} กำลังหมุน {curr_wheel.get('display')} "
                    f"แต่ช่องขวา #{right_wheel.get('pos', i+2)} เพิ่งอยู่ที่เลข '{val_right}' (ขัดแย้งฟันเฟือง)"
                )

    return errors


def validate_meter(results, min_conf=0.60):
    """
    ฟังก์ชันหลัก: รวมการตรวจทั้ง 2 กฎเข้าด้วยกัน
    Return: (is_valid: bool, errors: list)
    """
    errors = []
    errors.extend(check_confidence(results, min_conf=min_conf))
    errors.extend(check_gear_consistency(results))

    is_valid = (len(errors) == 0)
    return is_valid, errors


if __name__ == "__main__":
    # ทดสอบกรณีตัวเลขปกติ (Pass)
    test_pass = [
        {"pos": 1, "val_after": "0", "confidence": 0.95, "is_transition": False, "display": "0"},
        {"pos": 2, "val_after": "4", "confidence": 0.88, "is_transition": False, "display": "4"},
        {"pos": 3, "val_after": "2", "confidence": 0.91, "is_transition": False, "display": "2"},
        {"pos": 4, "val_after": "8", "confidence": 0.85, "is_transition": False, "display": "8"},
    ]
    is_valid, errors = validate_meter(test_pass)
    print("ทดสอบปกติ:", "✅ PASS" if is_valid else "❌ FAIL", errors)

    # ทดสอบกรณีขัดแย้งฟันเฟือง (Fail)
    test_gear_fail = [
        {"pos": 1, "val_after": "0", "confidence": 0.95, "is_transition": False, "display": "0"},
        {"pos": 2, "val_after": "5", "confidence": 0.75, "is_transition": True, "display": "[4->5]"},
        {"pos": 3, "val_after": "4", "confidence": 0.90, "is_transition": False, "display": "4"},
    ]
    is_valid, errors = validate_meter(test_gear_fail)
    print("ทดสอบขัดแย้งฟันเฟือง:", "✅ PASS" if is_valid else "❌ FAIL", errors)
