"""Gemini Verifier: Secondary AI Verification for Ambiguous or Disputed Meter Readings.

Uses Google Gemini Vision (Multimodal) to inspect the meter photo when Local YOLO/CNN
fails rule-based validation (Low confidence, gear conflict, or anomaly).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class GeminiVerifier:
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name

    def is_available(self) -> bool:
        """Check if Gemini API Key is configured."""
        return bool(self.api_key)

    def verify_image(self, image_path: str, context_notes: str = "") -> Dict[str, Any]:
        """Send the meter image and rule-based failure notes to Gemini Vision for arbitration."""
        image_path = Path(image_path)
        if not image_path.exists():
            return {"success": False, "error": f"Image not found: {image_path}"}

        if not self.is_available():
            return {
                "success": False,
                "error": "GEMINI_API_KEY not set. Please set the environment variable GEMINI_API_KEY.",
                "human_review_required": True,
            }

        prompt = f"""You are an expert utility meter inspector (Electric, Water, Gas meters).
An automated computer vision system tried to read this meter image, but encountered uncertainties:
[System Notes]: {context_notes}

Please carefully inspect the meter dials/wheels from left to right:
1. Identify all digit positions from left to right.
2. For each wheel, determine the exact digit (0-9).
3. If a wheel is halfway/transitioning between two digits (e.g. 4 rolling into 5), report the transition state.
4. Note which dials are red/decimal digits if applicable.

Return ONLY a JSON response in the following format (no markdown code blocks, just raw json):
{{
  "meter_reading_raw": "01234",
  "digits": [
    {{"pos": 1, "digit": "0", "is_transition": false, "confidence": 0.95}},
    {{"pos": 2, "digit": "1", "is_transition": false, "confidence": 0.95}}
  ],
  "decimal_count": 0,
  "confidence_overall": 0.95,
  "reasoning": "Clear view of dials, no blur, numbers 0 1 2 3 4 are stable."
}}
"""

        try:
            # Try new google-genai SDK first
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.model_name,
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
                # Fallback to google-generativeai legacy SDK
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.model_name)
                
                import PIL.Image
                img = PIL.Image.open(image_path)
                response = model.generate_content([prompt, img])
                text = response.text.strip()

            # Clean markdown code block if present
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()

            parsed = json.loads(text)
            return {"success": True, "data": parsed, "raw_response": text}

        except Exception as e:
            return {
                "success": False,
                "error": f"Gemini API request failed: {str(e)}",
                "human_review_required": True
            }


if __name__ == "__main__":
    verifier = GeminiVerifier()
    print("Gemini Available:", verifier.is_available())
    if not verifier.is_available():
        print("💡 Hint: Set GEMINI_API_KEY environment variable to test live API calls.")
