"""Meter Validator: Rule-Based Validation Engine for Industrial / Utility Meter Readings.

Rules implemented:
1. check_confidence: ตรวจสอบความมั่นใจของตัวเลขแต่ละหลัก (Confidence Threshold)
2. check_gear_consistency: ตรวจสอบความสัมพันธ์ของฟันเฟืองมิเตอร์ (Mechanical Gear Consistency)
3. check_reading_not_decreased: ค่าน้ำ/ไฟต้องไม่ลดลง (เดือนนี้ต้อง >= เดือนที่แล้ว)
4. check_usage_not_abnormal: อัตราการใช้ต้องไม่ผิดปกติ (เทียบกับค่าเฉลี่ย 3 เดือน)
5. validate_meter: ฟังก์ชันหลักที่รวมการตรวจทุกกฎเข้าด้วยกัน
"""

import sys

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# กำหนดจำนวนหลักมาตรฐานของมิเตอร์แต่ละชนิด
VALID_DIGIT_COUNTS = {
    "elec":  {5},            # ไฟฟ้าต้อง 5 หลัก
    "gas":   {8},            # แก๊สต้อง 8 หลัก
    "water": {4, 5, 7, 8},   # น้ำยอมรับได้ทั้ง 4, 5, 7, 8 หลัก
}


def check_confidence(results, min_conf=0.60, meter_type="auto"):
    """
    กฎข้อที่ 1: ตรวจสอบความมั่นใจและความสมบูรณ์ของตัวเลข (Confidence & Completeness)
    - จำนวนหลักต้องตรงตามมาตรฐานของมิเตอร์ประเภทนั้นๆ
    - ต้องไม่มีตัว '?'
    - ค่าความมั่นใจต้องไม่ต่ำกว่า min_conf
    """
    errors = []

    # 1.1 ตรวจสอบจำนวนหลักตามประเภทมิเตอร์
    valid_set = VALID_DIGIT_COUNTS.get(meter_type)
    if valid_set and len(results) not in valid_set:
        errors.append(
            f"[Rule 1] จำนวนหลักไม่ถูกต้อง: ตรวจพบ {len(results)} หลัก "
            f"(มิเตอร์ประเภท {meter_type} ต้องมี {sorted(list(valid_set))} หลัก)"
        )
    elif meter_type == "water" and len(results) == 4:
        # ดักเคสพิเศษ: มิเตอร์น้ำ 4 หลักแท้ต้องเป็นดำล้วน ถ้ามีล้อสีแดงปนมา แสดงว่าอ่านขาดจากรุ่น 5 หลัก
        if any(r.get("is_red", False) for r in results):
            errors.append(
                f"[Rule 1] โครงสร้างหลักผิดปกติ: ตรวจพบ 4 หลักแต่มีล้อสีแดงปนมา (น่าจะอ่านขาดจากรุ่น 5 หลัก)"
            )

    for r in results:
        pos = r.get("pos", "?")
        val = r.get("val_after", "?")
        conf = float(r.get("confidence", 0.0))

        if val == "?" or not str(val).isdigit():
            errors.append(f"[Rule 1] ช่องที่ {pos} อ่านค่าตัวเลขไม่ชัดเจน (ได้ค่า '{val}')")
        elif conf < min_conf:
            errors.append(f"[Rule 1] ช่องที่ {pos} ความมั่นใจต่ำ ({conf:.2f} < {min_conf:.2f})")

    return errors


# ===================================================================
# กฎข้อที่ 2: ค่าน้ำ/ไฟต้องไม่ลดลง (ประวัติเทียบค่าปัจจุบัน)
# ===================================================================
def check_reading_not_decreased(current_reading: float, history: list) -> list:
    """
    กฎข้อที่ 2: ค่ามิเตอร์ต้องไม่ลดลง
    - ค่าของเดือนนี้ต้องมากกว่าหรือเท่ากับเดือนที่แล้ว
    - history = ลิสต์ค่าย้อนหลัง เรียงจากเก่าไปใหม่ เช่น [1200.5, 1250.0, 1310.2]
      ตัวสุดท้าย (history[-1]) คือค่าเดือนที่แล้ว
    """
    errors = []
    if not history:
        return errors

    last_reading = history[-1]
    if current_reading < last_reading:
        errors.append(
            f"[Rule 2] ค่ามิเตอร์ลดลงผิดปกติ: "
            f"ค่าอ่านได้ = {current_reading:.1f} แต่เดือนที่แล้ว = {last_reading:.1f} "
            f"(ค่ามิเตอร์ถอยหลังไม่ได้)"
        )

    return errors


# ===================================================================
# กฎข้อที่ 3: อัตราการใช้ต้องไม่ผิดปกติ (เทียบค่าเฉลี่ย 3 เดือน)
# ===================================================================
USAGE_SPIKE_MULTIPLIER = float(__import__('os').getenv("USAGE_SPIKE_MULTIPLIER", "3.0"))


def check_usage_not_abnormal(current_reading: float, history: list, spike_multiplier: float = None) -> list:
    """
    กฎข้อที่ 3: ตรวจสอบว่าอัตราการใช้ไม่พุ่งผิดปกติ
    - คำนวณหน่วยที่ใช้แต่ละเดือนจากประวัติ 3 เดือน
    - หาค่าเฉลี่ยการใช้ต่อเดือน
    - ถ้าเดือนนี้ใช้เกินค่าเฉลี่ย x เท่า (ค่าเริ่มต้น 3 เท่า) → ผิดปกติ
    """
    if spike_multiplier is None:
        spike_multiplier = USAGE_SPIKE_MULTIPLIER

    errors = []
    if not history or len(history) < 2:
        return errors

    # คำนวณหน่วยที่ใช้แต่ละเดือนจากประวัติ
    monthly_usages = []
    for i in range(1, len(history)):
        usage = history[i] - history[i - 1]
        if usage >= 0:
            monthly_usages.append(usage)

    if not monthly_usages:
        return errors

    avg_usage = sum(monthly_usages) / len(monthly_usages)
    current_usage = current_reading - history[-1]

    # ถ้าค่าเฉลี่ยเป็น 0 (ไม่มีการใช้มา 3 เดือน) แต่เดือนนี้มีการใช้ → แจ้งเตือน
    if avg_usage == 0 and current_usage > 0:
        errors.append(
            f"[Rule 3] ผิดปกติ: 3 เดือนก่อนไม่มีการใช้งานเลย (เฉลี่ย = 0) "
            f"แต่เดือนนี้ใช้ไป {current_usage:.1f} หน่วย"
        )
        return errors

    if avg_usage > 0 and current_usage > avg_usage * spike_multiplier:
        errors.append(
            f"[Rule 3] อัตราการใช้ผิดปกติ: "
            f"เดือนนี้ใช้ไป {current_usage:.1f} หน่วย "
            f"แต่ค่าเฉลี่ย 3 เดือน = {avg_usage:.1f} หน่วย/เดือน "
            f"(เกิน {spike_multiplier:.0f} เท่าของค่าเฉลี่ย)"
        )

    return errors


# ===================================================================
# ฟังก์ชันหลัก: รวมทุกกฎเข้าด้วยกัน (3 กฎมาตรฐาน)
# ===================================================================
def validate_meter(results, min_conf=0.60, current_reading: float = None, history: list = None, meter_type: str = "auto"):
    """
    ฟังก์ชันหลัก: รวมการตรวจ 3 กฎมาตรฐานเข้าด้วยกัน
    - Rule 1: ความชัดเจนและความสมบูรณ์ของตัวเลข (Confidence & Completeness Check)
    - Rule 2: ค่ามิเตอร์ต้องไม่ลดลง (Reading Not Decreased)
    - Rule 3: อัตราการใช้ต้องไม่พุ่งผิดปกติ (Usage Anomaly Check)
    Return: (is_valid: bool, errors: list)
    """
    errors = []

    # ถ้าไม่พบช่องตัวเลขเลย ให้ถือว่าไม่ผ่านทันที
    if not results:
        return False, ["ไม่พบล้อตัวเลขในภาพเลย"]

    # Rule 1: ตรวจความมั่นใจและความสมบูรณ์ของตัวเลข (Confidence & Completeness)
    errors.extend(check_confidence(results, min_conf=min_conf, meter_type=meter_type))

    # Rule 2 & 3: ตรวจจากประวัติ (ถ้ามี)
    if current_reading is not None and history:
        errors.extend(check_reading_not_decreased(current_reading, history))
        errors.extend(check_usage_not_abnormal(current_reading, history))

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

    print("\n" + "=" * 60)
    print("ทดสอบกฎข้อ 3 & 4 (ประวัติ 3 เดือน)")
    print("=" * 60)

    test_results = [
        {"pos": 1, "val_after": "1", "confidence": 0.95, "is_transition": False, "display": "1"},
        {"pos": 2, "val_after": "3", "confidence": 0.90, "is_transition": False, "display": "3"},
        {"pos": 3, "val_after": "5", "confidence": 0.88, "is_transition": False, "display": "5"},
        {"pos": 4, "val_after": "0", "confidence": 0.85, "is_transition": False, "display": "0"},
    ]
    history = [1200.0, 1250.0, 1310.0]

    # ทดสอบค่าปกติ (1350 > 1310 และอัตราการใช้ 40 ไม่เกิน 3 เท่าของเฉลี่ย 55)
    is_valid, errors = validate_meter(test_results, current_reading=1350.0, history=history)
    print(f"\nค่าปกติ (1350): {'✅ PASS' if is_valid else '❌ FAIL'}")
    for e in errors:
        print(f"  {e}")

    # ทดสอบค่าลดลง (1100 < 1310 → Rule 3 Fail)
    is_valid, errors = validate_meter(test_results, current_reading=1100.0, history=history)
    print(f"\nค่าลดลง (1100): {'✅ PASS' if is_valid else '❌ FAIL'}")
    for e in errors:
        print(f"  {e}")

    # ทดสอบอัตราการใช้ผิดปกติ (1600 - 1310 = 290, เฉลี่ย 55, เกิน 3 เท่า → Rule 4 Fail)
    is_valid, errors = validate_meter(test_results, current_reading=1600.0, history=history)
    print(f"\nอัตราผิดปกติ (1600): {'✅ PASS' if is_valid else '❌ FAIL'}")
    for e in errors:
        print(f"  {e}")
