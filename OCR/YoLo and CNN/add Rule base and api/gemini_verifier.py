"""Gemini Verifier: Secondary AI Verification for Ambiguous Meter Readings.

ส่งรูปภาพมิเตอร์ไปถาม Gemini Vision เมื่อ Local AI ไม่ผ่านเกณฑ์ Rule-Base
"""

import os
import sys
import json
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def verify_with_gemini(image_path: str, error_notes: str = "", api_key: str = None) -> dict:
    """
    ส่งภาพมิเตอร์และข้อความแจ้งปัญหาไปให้ Gemini Vision ช่วยอ่านซ้ำ
    Return: dict {"success": bool, "reading": str, "reason": str, "error": str}
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY") or "AQ.Ab8RN6LM3BYwtLAAp_t7-0h8xCTyeHvyKRlXjCXlZkjo54Fi-A"
    if not api_key:
        return {"success": False, "error": "ไม่ได้ตั้งค่า GEMINI_API_KEY"}

    image_path = Path(image_path)
    if not image_path.exists():
        return {"success": False, "error": f"ไม่พบไฟล์ภาพ: {image_path}"}

    prompt = f"""You are an expert utility meter inspector (Electric, Water, Gas meters).
Our automated computer vision system tried to read this meter image, but encountered uncertainties:
[Issues]: {error_notes}

Please carefully inspect the meter dials/wheels from left to right:
1. Read each digit from left to right.
2. Return ONLY valid JSON format without markdown code blocks.

Format:
{{
  "meter_reading": "03548",
  "reasoning": "Dials are clearly visible. Values are 0 3 5 4 8."
}}
"""

    try:
        # พยายามใช้ google-genai หรือ fallback เป็น google-generativeai
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    genai.types.Part.from_bytes(
                        data=image_path.read_bytes(),
                        mime_type="image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
                    ),
                    prompt
                ]
            )
            text = response.text.strip()
        except ImportError:
            import google.generativeai as genai
            import PIL.Image
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-3.6-flash")
            img = PIL.Image.open(image_path)
            response = model.generate_content([prompt, img])
            text = response.text.strip()

        # ลบ markdown code block ออก
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        return {
            "success": True,
            "reading": str(data.get("meter_reading", "")),
            "reason": str(data.get("reasoning", ""))
        }

    except Exception as e:
        return {"success": False, "error": f"Gemini Error: {str(e)}"}


if __name__ == "__main__":
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("💡 Hint: ยังไม่ได้ตั้งค่า GEMINI_API_KEY ใน Environment")
    else:
        print("✅ GEMINI_API_KEY พร้อมใช้งาน")
