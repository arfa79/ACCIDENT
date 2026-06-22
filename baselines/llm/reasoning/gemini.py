"""
Gemini-based VLM reasoner for accident detection.

Mirrors the interface of `QwenVLReasoner` so the existing temporal/spatial pipeline
can use it as a drop-in replacement. Uses the Google Gemini API (no local GPU).

Set the API key via the GEMINI_API_KEY environment variable.
"""

import io
import os
import re
import time
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types

from reasoning.utils import match_to_class


DEFAULT_MODEL = "gemini-2.0-flash"


def _pil_to_part(img: Image.Image) -> types.Part:
    """Convert PIL image to a Gemini inline Part."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


class GeminiVLReasoner:
    def __init__(self, model_name: str = DEFAULT_MODEL, requests_per_minute: int = 14):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before instantiating GeminiVLReasoner")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self._min_interval = 60.0 / max(1, requests_per_minute)
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call = time.monotonic()

    def generate_text(self, img: Image.Image, prompt: str, max_retries: int = 5) -> str:
        """Generate text for a given image and prompt, with retry on transient errors."""
        for attempt in range(max_retries):
            self._throttle()
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[_pil_to_part(img), prompt],
                )
                return (response.text or "").strip()
            except Exception as e:
                msg = str(e).lower()
                if "rate" in msg or "quota" in msg or "503" in msg or "500" in msg:
                    time.sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError(f"Gemini call failed after {max_retries} retries")

    def accident_temporal_reasoning(self, imgs, prompt=None):
        """
        Mirrors QwenVLReasoner.accident_temporal_reasoning:
        - imgs is a list of (frame_id, PIL.Image)
        - returns (first_yes_frame_id_or_None, raw_predictions_dict)
        """
        if prompt is None:
            prompt = (
                "Is there a traffic accident or collision? "
                "Yes or No answer only."
            )

        preds_raw = {}
        for i, img in imgs:
            preds_raw[i] = self.generate_text(img, prompt)

        preds = {k: "yes" in v.strip().lower() for k, v in preds_raw.items()}
        preds = pd.Series(preds)
        frame_ids = preds[preds].index
        frame_id = frame_ids[0] if len(frame_ids) else None
        return frame_id, preds_raw

    def accident_spatial_reasoning(self, img, prompt=None):
        if prompt is None:
            prompt = (
                "The scene depicts a traffic accident with one or more cars colliding. "
                "Where is the accident? Reply with normalized coordinates as: "
                "<point x='0.0-1.0' y='0.0-1.0'>accident</point>"
            )
        generated_text = self.generate_text(img, prompt)
        point = self.parse_point(generated_text, img.size)
        return point, generated_text

    def accident_cause_reasoning(self, img, prompt):
        generated_text = self.generate_text(img, prompt)
        label = match_to_class(generated_text)
        return label, generated_text

    @staticmethod
    def parse_point(text: str, image_size: tuple[int, int]) -> tuple[int | None, int | None]:
        """
        Parse the first <point x='...' y='...'> tag. Accepts both normalized (0-1)
        and pixel coordinates and returns pixel coordinates.
        """
        match = re.search(r"<point[^>]*x=['\"]([\d.]+)['\"][^>]*y=['\"]([\d.]+)['\"]", text)
        if not match:
            return (None, None)

        x, y = float(match.group(1)), float(match.group(2))
        w, h = image_size
        if x <= 1.0 and y <= 1.0:
            x, y = x * w, y * h
        return (int(x), int(y))
