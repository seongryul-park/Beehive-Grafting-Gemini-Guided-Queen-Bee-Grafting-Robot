"""
cell_detector.py — Stage 1: locate individual honeycomb cells.

detect(image_path) -> {
    "width": W, "height": H,
    "cells": [{"cell_id": 1, "bbox": [x1, y1, x2, y2]}, ...]   # pixel coords
}

Backends (DETECTOR env):
  - "gemini": one Gemini call returns boxes for all visible cells
  - "geometry"/"opencv": honeycomb CELL detector in opencv_prototype (wax-wall geometry)
  - "color":  offline color-segmentation candidate generator in color_segment
              (recall-oriented; feeds Gemini biological verification)

Mock mode scatters believable, irregular boxes over the actual image so the
dashboard can be demoed with no API call.
"""

import json
import re
import random

from PIL import Image


class DetectorError(Exception):
    """Raised when detection fails or returns unusable output."""


def coerce_boxes(data, W, H):
    """Accept [{bbox:[...]}|[...]] items; rescale if model used 0-1 or 0-1000."""
    raw = []
    for item in data:
        if isinstance(item, dict):
            b = item.get("bbox") or item.get("box") or item.get("rect")
        else:
            b = item
        if isinstance(b, (list, tuple)) and len(b) == 4:
            try:
                raw.append([float(v) for v in b])
            except (TypeError, ValueError):
                continue
    if not raw:
        return []
    maxv = max(max(abs(v) for v in b) for b in raw)
    if maxv <= 1.5:               # normalized 0-1
        sx, sy = W, H
    elif maxv > max(W, H) * 1.2:  # Gemini's 0-1000 convention
        sx, sy = W / 1000.0, H / 1000.0
    else:                          # already pixels
        sx = sy = 1.0
    return [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in raw]


def clamp_box(b, W, H):
    """Order corners, clamp to image bounds, drop degenerate boxes."""
    try:
        x1, y1, x2, y2 = (int(round(v)) for v in b)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0, min(x1, W)); x2 = max(0, min(x2, W))
    y1 = max(0, min(y1, H)); y2 = max(0, min(y2, H))
    if (x2 - x1) < 4 or (y2 - y1) < 4:
        return None
    return [x1, y1, x2, y2]


_DET_PROMPT = (
    "This is a {W}x{H} pixel photo of a honeycomb. Detect EVERY visible cell "
    "opening (the individual hexagonal/round wax cells), including partially "
    "visible ones at the edges.\n"
    "Respond with STRICT JSON only — no markdown, no prose — as an array of "
    'objects, one per cell: [{{"bbox": [x1, y1, x2, y2]}}] where the coordinates '
    "are PIXELS in this image, origin top-left, x in [0,{W}], y in [0,{H}], "
    "and (x1,y1) is the top-left, (x2,y2) the bottom-right corner."
)


class CellDetector:
    def __init__(self, method="gemini", client=None, model="", mock=False, max_cells=24):
        self.method = method
        self.client = client
        self.model = model
        self.mock = mock
        self.max_cells = max_cells

    # ---- public API --------------------------------------------------------

    def detect(self, image_path: str) -> dict:
        with Image.open(image_path) as im:
            width, height = im.size

        # Offline detectors (color/opencv) need no Gemini, so they ALWAYS run for
        # real — even in mock. `mock` must not replace real candidates with fake
        # boxes; it only stands in for the Gemini *detection* call (DETECTOR=gemini).
        # This keeps the OpenCV candidates connected to (mock or live) Gemini.
        if self.method in ("opencv", "geometry"):
            boxes = self._detect_opencv(image_path, width, height)
        elif self.method == "color":
            boxes = self._detect_color(image_path, width, height)
        elif self.mock:
            boxes = self._mock_boxes(width, height)
        else:
            boxes = self._detect_gemini(image_path, width, height)

        # clamp, drop degenerate boxes, cap count, assign ids
        cells = []
        for b in boxes:
            box = clamp_box(b, width, height)
            if box is not None:
                cells.append(box)
            if len(cells) >= self.max_cells:
                break

        if not cells:
            raise DetectorError("no cells detected")

        return {
            "width": width,
            "height": height,
            "cells": [{"cell_id": i + 1, "bbox": b} for i, b in enumerate(cells)],
        }

    # ---- Gemini detection --------------------------------------------------

    def _detect_gemini(self, image_path, W, H):
        from google.genai import types  # imported lazily so opencv/mock need no SDK

        img = Image.open(image_path)
        img.load()
        prompt = _DET_PROMPT.format(W=W, H=H)
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.0
                ),
            )
        except Exception as e:
            raise DetectorError(f"Gemini detection failed: {e}") from e

        text = getattr(resp, "text", None)
        if not text:
            raise DetectorError("Gemini detection returned empty response")
        data = self._parse_array(text)
        return coerce_boxes(data, W, H)

    # ---- OpenCV detection (offline) ---------------------------------------

    def _detect_opencv(self, image_path, W, H):
        """Stage-1 OpenCV layer: locate cells offline, then feed Stage 2 classify.

        Delegates to opencv_prototype (HoughCircles + adaptive-threshold contour
        centroids, then aggressive center-distance de-dup -> one center per cell).
        Imported lazily so the gemini/mock paths need neither cv2 nor numpy.
        """
        try:
            import opencv_prototype
        except ImportError as e:
            raise DetectorError(
                "opencv detector needs opencv-python-headless + numpy"
            ) from e

        try:
            result = opencv_prototype.detect(image_path, max_cells=self.max_cells * 4)
        except ValueError as e:  # unreadable image, no cells, etc.
            raise DetectorError(str(e)) from e

        boxes = [c["bbox"] for c in result.get("cells", [])]
        if not boxes:
            raise DetectorError(
                "opencv found no cells (try DETECTOR=gemini or tune params)"
            )
        return boxes

    # ---- color-segmentation detection (offline candidate generator) -------

    def _detect_color(self, image_path, W, H):
        """Recall-oriented candidate proposer for the Gemini-verification path.

        Delegates to color_segment (HSV/LAB bright-pale brood mask -> connected
        components). Intentionally over-proposes (reflections, wax, eggs, empty
        cells); Gemini does the biological verification downstream. Imported
        lazily so the gemini/mock paths need neither cv2 nor numpy.
        """
        try:
            import color_segment
        except ImportError as e:
            raise DetectorError(
                "color detector needs opencv-python-headless + numpy"
            ) from e

        try:
            result = color_segment.detect(image_path, max_cells=self.max_cells * 4)
        except ValueError as e:  # unreadable image, no candidates, etc.
            raise DetectorError(str(e)) from e

        boxes = [c["bbox"] for c in result.get("cells", [])]
        if not boxes:
            raise DetectorError("color segmentation found no candidate regions")
        return boxes

    # ---- mock detection ----------------------------------------------------

    def _mock_boxes(self, W, H):
        """Deterministic, irregular scatter (staggered + jittered, varied sizes)."""
        rng = random.Random(42)
        rows, cols = 3, 4
        base = max(16, int(min(W, H) / 6))
        boxes = []
        for r in range(rows):
            stagger = (base // 2) if (r % 2) else 0
            for c in range(cols):
                cx = int((c + 0.7) * (W / (cols + 0.6))) + stagger
                cy = int((r + 0.8) * (H / (rows + 0.6)))
                cx += rng.randint(-base // 6, base // 6)
                cy += rng.randint(-base // 6, base // 6)
                s = base + rng.randint(-base // 5, base // 5)
                boxes.append([cx - s // 2, cy - s // 2, cx + s // 2, cy + s // 2])
        rng.shuffle(boxes)  # so cell_ids aren't in tidy reading order
        return boxes

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _parse_array(text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                raise DetectorError(f"could not parse detection JSON: {text[:200]!r}")
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                raise DetectorError(f"could not parse detection JSON: {text[:200]!r}")
        if not isinstance(data, list):
            raise DetectorError("detection output is not a JSON array")
        return data
