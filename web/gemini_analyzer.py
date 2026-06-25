"""
gemini_analyzer.py — Gemini biological CLASSIFICATION (Stage 3).

The image goes: OpenCV cell detection -> one cell = one crop -> OpenCV fast
visual filter -> THIS stage. Gemini receives the original image (context) plus
ONE single-cell crop per surviving candidate, and ONLY classifies each supplied
crop. It does NOT detect cells or search the image.

BATCHING: sending 100+ crops in a single multimodal request was unstable
(empty responses, OTHER-blocked prompts). The crops are now split into batches
of ~32-48 (BATCH_SIZE) and each batch is one Gemini call. cell_id is preserved
across batches and every batch's output is merged into one unified result before
ranking. Batches are independent: if one fails, the others still count and the
cells of the failed batch degrade to a safe 'empty' instead of losing the whole
image.

RAG REFERENCE GALLERY: a small, manually curated set of close-ups of an IDEAL
young larva (sample_images/rag/rag*.*) is loaded once and prepended to EVERY
batch as calibration-only, few-shot exemplars BEFORE the original image and the
candidate crops. Gemini is told not to classify them, only to judge candidates
by similarity to them. If the folder is absent the prompt falls back to the
original no-reference behaviour.

For each crop Gemini returns:
  {cell_id, classification, confidence, graft_score, reason}
  classification ∈ egg | young_larva | old_larva | empty
  confidence   [0,1] certainty in the classification
  graft_score  [0,1] grafting suitability, INDEPENDENT of confidence
  reason       short justification

Strict JSON requested + validated; a bad cell degrades to a safe 'empty'
(graft_score 0) instead of failing the image. Mock makes no call.
"""

import glob
import json
import os
import re

from PIL import Image
from google.genai import types

from cell_detector import coerce_boxes, clamp_box


class AnalyzerError(Exception):
    """Raised when the Gemini call itself fails or returns nothing usable."""


CLASSES = ("egg", "young_larva", "old_larva", "empty")

CROP_PAD = 4    # small extra margin around each already-single-cell box
BATCH_SIZE = 48  # target max crops per Gemini request (~32-48 is the safe range)

# RAG reference gallery: manually curated close-ups of an IDEAL young larva for
# grafting. Loaded once and prepended (calibration-only) to every Gemini batch.
# Default folder + filename pattern; comb.jpg (a whole comb) is intentionally
# excluded so only single-cell exemplars become references.
REF_DIR_DEFAULT = os.path.join(os.path.dirname(__file__), "sample_images", "rag")
REF_PATTERNS = ("rag*.jpg", "rag*.jpeg", "rag*.png")

_FIELD_DEFS = (
    "  classification: exactly one of —\n"
    "    - egg:         tiny rice-grain egg standing at the cell base\n"
    "    - young_larva: small C-shaped larva in royal jelly (~12-24h, IDEAL to graft)\n"
    "    - old_larva:   large plump curled larva filling much of the cell (too old)\n"
    "    - empty:       drawn comb, no egg or larva (also use for wax/reflection/unclear)\n"
    "  confidence:     0..1 certainty in the classification.\n"
    "  graft_score:    0..1 suitability for queen GRAFTING, INDEPENDENT of\n"
    "                  confidence. Highest for a young_larva (~12-24h); near 0 for\n"
    "                  old_larva, egg, or empty.\n"
    "  reason:         short (<=15 words) justification.\n"
)

_JSON_TAIL = (
    "Respond with STRICT JSON only — no markdown, no prose — exactly one object "
    "per CANDIDATE crop, covering every cell_id in {ids}: "
    '[{{"cell_id": <int>, "classification": "<enum>", "confidence": <float 0..1>, '
    '"graft_score": <float 0..1>, "reason": "<short>"}}]'
)

# No reference gallery available — original behaviour.
_CLASSIFY_PROMPT = (
    "An upstream OpenCV pipeline has ALREADY detected the honeycomb cells and "
    "pre-filtered them. The FIRST image is the full original honeycomb photo, for "
    "global context. Each SUBSEQUENT image is a cropped close-up of ONE honeycomb "
    "cell. The crops are supplied in this cell_id order: {ids}.\n"
    "\n"
    "Your job is biological CLASSIFICATION only:\n"
    "  - DO NOT detect cells or search the whole honeycomb.\n"
    "  - DO NOT skip or ignore any supplied crop.\n"
    "  - Treat each crop as exactly one already-isolated cell and classify it.\n"
    "\n"
    "Act as an apiculture expert. For EACH supplied crop (by its cell_id) return:\n"
    + _FIELD_DEFS + _JSON_TAIL
)

# Reference gallery present — prepend curated young-larva exemplars as a
# calibration anchor (in-context, few-shot) BEFORE the original image and crops.
_CLASSIFY_PROMPT_REF = (
    "An upstream OpenCV pipeline has ALREADY detected the honeycomb cells and "
    "pre-filtered them. You receive images in THREE groups, in this exact order:\n"
    "  1) {n_ref} REFERENCE image(s): each is a confirmed close-up of an IDEAL "
    "young larva for queen grafting — a small C-shaped larva (~12-24h) curled in "
    "glistening royal jelly at the cell base. These are GROUND TRUTH for the "
    "'young_larva' class. Study them to calibrate your eye. DO NOT classify or "
    "return any object for the reference images.\n"
    "  2) ONE full original honeycomb photo, for global context.\n"
    "  3) The CANDIDATE crops to classify — one isolated cell each — supplied in "
    "this cell_id order: {ids}.\n"
    "\n"
    "Your job is biological CLASSIFICATION of the CANDIDATE crops only:\n"
    "  - DO NOT detect cells or search the whole honeycomb.\n"
    "  - DO NOT skip or ignore any candidate crop.\n"
    "  - Judge each candidate by visual similarity to the reference young larvae: "
    "the closer the size, C-shape, and fresh royal jelly match, the stronger the "
    "case for 'young_larva' (and a high graft_score). A larva clearly LARGER / "
    "plumper than the references is 'old_larva'; a tiny upright grain is 'egg'; a "
    "bare cell is 'empty'.\n"
    "  - Treat each candidate crop as exactly one already-isolated cell.\n"
    "\n"
    "Act as an apiculture expert. For EACH candidate crop (by its cell_id) return:\n"
    + _FIELD_DEFS + _JSON_TAIL
)

_COMBINED_PROMPT = (
    "This is a {W}x{H} pixel photo of a honeycomb. In ONE pass, detect EVERY "
    "visible cell opening (including partially visible ones) and, acting as an "
    "apiculture expert, classify each. Fields:\n" + _FIELD_DEFS +
    "Number cells sequentially with cell_id starting at 1.\n"
    "Respond with STRICT JSON only — no markdown, no prose — one object per cell: "
    '[{{"cell_id": <int starting at 1>, "bbox": [x1, y1, x2, y2], '
    '"classification": "<enum>", "confidence": <float 0..1>, '
    '"graft_score": <float 0..1>, "reason": "<short>"}}] where bbox is in PIXELS, '
    "origin top-left, x in [0,{W}], y in [0,{H}]."
)


class GeminiAnalyzer:
    def __init__(self, client=None, model="", mock=False, max_cells=150,
                 batch_size=BATCH_SIZE, reference_dir=REF_DIR_DEFAULT):
        self.client = client
        self.model = model
        self.mock = mock
        self.max_cells = max_cells
        self.batch_size = max(1, int(batch_size))
        self.reference_dir = reference_dir
        self._ref_cache = None  # lazily loaded list of PIL.Image references

    # ---- RAG reference gallery (curated ideal-young-larva exemplars) --------

    def _references(self):
        """Load + cache the curated young-larva reference crops (once).

        Returns a list of RGB PIL images. Missing/empty/unreadable folder ->
        [] so classification silently falls back to the no-reference prompt.
        """
        if self._ref_cache is not None:
            return self._ref_cache
        imgs = []
        d = self.reference_dir
        if d and os.path.isdir(d):
            paths = []
            for pat in REF_PATTERNS:
                paths.extend(glob.glob(os.path.join(d, pat)))
            for p in sorted(set(paths)):
                if os.path.basename(p).lower().startswith("comb"):
                    continue  # whole-comb image is not a single-cell exemplar
                try:
                    im = Image.open(p)
                    im.load()
                    imgs.append(im.convert("RGB"))
                except Exception:
                    pass  # skip an unreadable reference, keep the rest
        self._ref_cache = imgs
        return imgs

    # ---- batched classification of the supplied single-cell crops -----------

    def classify_cells(self, image_path: str, cells: list) -> dict:
        """Classify every supplied crop, splitting them across Gemini calls.

        Returns the usual {width, height, cells} plus batching metadata:
          gemini_calls  : number of Gemini requests made (0 in mock)
          batches       : [{index, size, ok, error?}] one per batch
          failed_batches: subset of `batches` that raised (size + error)
        Every input cell appears in `cells`; cells whose batch failed (or that
        Gemini never returned) degrade to a safe 'empty' verdict.
        """
        with Image.open(image_path) as im:
            W, H = im.size

        if self.mock:
            out = [{**c, **self._mock(c["cell_id"] - 1)} for c in cells]
            return {"width": W, "height": H, "cells": out,
                    "gemini_calls": 0, "batches": [], "failed_batches": []}

        if not cells:
            return {"width": W, "height": H, "cells": [],
                    "gemini_calls": 0, "batches": [], "failed_batches": []}

        img = Image.open(image_path)
        img.load()
        img = img.convert("RGB")

        # Number of batches is derived automatically from the candidate count:
        # fill batches of self.batch_size, leaving the remainder in the last.
        batches = [cells[i:i + self.batch_size]
                   for i in range(0, len(cells), self.batch_size)]

        by_id = {}
        batch_info = []
        failed = []
        calls = 0
        for idx, batch in enumerate(batches, start=1):
            calls += 1
            try:
                data = self._classify_batch(img, batch, W, H)
                for d in data:
                    if isinstance(d, dict) and "cell_id" in d:
                        try:
                            by_id[int(d["cell_id"])] = d
                        except (TypeError, ValueError):
                            pass
                batch_info.append({"index": idx, "size": len(batch), "ok": True})
            except AnalyzerError as e:
                # A single batch failing must NOT lose the whole image: record it
                # and keep going so the other batches' results survive.
                info = {"index": idx, "size": len(batch), "ok": False,
                        "error": str(e)}
                batch_info.append(info)
                failed.append({"index": idx, "size": len(batch), "error": str(e)})

        # Merge: every input cell, in original order, gets its verdict (or a safe
        # 'empty' if its batch failed / Gemini omitted it).
        out = [{**c, **self._validate_one(by_id.get(c["cell_id"], {}))}
               for c in cells]
        return {"width": W, "height": H, "cells": out,
                "gemini_calls": calls, "batches": batch_info,
                "failed_batches": failed}

    def _classify_batch(self, img, batch, W, H):
        """One Gemini call for one batch of crops. Raises AnalyzerError on failure.

        Layout sent to Gemini: [prompt] -> [reference young-larva exemplars] ->
        [original honeycomb] -> [candidate crops]. The references are prepended
        to EVERY batch so each call is independently calibrated.
        """
        ids = [c["cell_id"] for c in batch]
        refs = self._references()

        contents = []
        if refs:
            contents.append(_CLASSIFY_PROMPT_REF.format(ids=ids, n_ref=len(refs)))
            contents.append(
                f"REFERENCE young-larva exemplars ({len(refs)}) — GROUND TRUTH "
                "for 'young_larva', for calibration only, DO NOT classify:"
            )
            contents.extend(refs)
        else:
            contents.append(_CLASSIFY_PROMPT.format(ids=ids))

        contents.append("ORIGINAL honeycomb image (global context):")
        contents.append(img)
        for c in batch:
            crop = self._crop(img, c["bbox"], W, H)
            contents.append(f"CANDIDATE CROP for cell_id {c['cell_id']}:")
            contents.append(crop)
        return self._parse_array(self._call(contents))

    @staticmethod
    def _crop(img, bbox, W, H, pad=CROP_PAD):
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - pad); y1 = max(0, int(y1) - pad)
        x2 = min(W, int(x2) + pad); y2 = min(H, int(y2) + pad)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, W, H
        return img.crop((x1, y1, x2, y2))

    # ---- one call: detect + classify together (DETECTOR=gemini) -------------

    def detect_and_classify(self, image_path: str) -> dict:
        img = Image.open(image_path)
        img.load()
        W, H = img.size
        prompt = _COMBINED_PROMPT.format(W=W, H=H)
        data = self._parse_array(self._call([prompt, img]))
        items = [
            it for it in data
            if isinstance(it, dict)
            and isinstance(it.get("bbox"), (list, tuple)) and len(it["bbox"]) == 4
        ]
        boxes = coerce_boxes([it["bbox"] for it in items], W, H)
        cells = []
        for it, b in zip(items, boxes):
            box = clamp_box(b, W, H)
            if box is None:
                continue
            cell = {"cell_id": len(cells) + 1, "bbox": box}
            cell.update(self._validate_one(it))
            cells.append(cell)
            if len(cells) >= self.max_cells:
                break
        if not cells:
            raise AnalyzerError("Gemini returned no usable cells")
        return {"width": W, "height": H, "cells": cells}

    # ---- Gemini call -------------------------------------------------------

    def _call(self, contents):
        # 2.5-flash spends output tokens on internal "thinking"; with a big
        # multimodal input + forced JSON it can burn the whole budget and return
        # NO text. Disable thinking and give a generous output budget so the JSON
        # actually comes back.
        cfg_kwargs = dict(
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=16384,
        )
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass  # older SDK without ThinkingConfig — still works, just may think

        try:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
        except Exception as e:
            raise AnalyzerError(f"Gemini request failed: {e}") from e

        text = getattr(resp, "text", None)
        if not text:
            finish = feedback = None
            try:
                finish = resp.candidates[0].finish_reason
            except Exception:
                pass
            try:
                feedback = resp.prompt_feedback
            except Exception:
                pass
            raise AnalyzerError(
                f"Gemini returned an empty response "
                f"(finish_reason={finish}, prompt_feedback={feedback}); "
                "this batch was rejected — try a smaller BATCH_SIZE."
            )
        return text

    # ---- mock (classification, confidence, graft_score, reason) ------------

    _MOCK = [
        ("empty",       0.95, 0.00, "clean drawn comb, nothing inside"),
        ("egg",         0.88, 0.35, "upright rice-grain egg at the cell base"),
        ("young_larva", 0.94, 0.95, "small C-shaped larva in fresh royal jelly"),
        ("old_larva",   0.80, 0.15, "plump curled larva, past grafting age"),
        ("empty",       0.92, 0.00, "empty drawn comb"),
        ("young_larva", 0.86, 0.88, "young larva, good size for grafting"),
        ("egg",         0.70, 0.30, "small egg, slightly blurred"),
        ("old_larva",   0.68, 0.12, "larva past ideal grafting window"),
        ("young_larva", 0.79, 0.82, "young larva, partially shadowed"),
        ("empty",       0.90, 0.00, "vacant cell"),
        ("egg",         0.83, 0.33, "egg leaning against the cell wall"),
        ("old_larva",   0.75, 0.10, "large larva, too developed"),
    ]

    def _mock(self, i):
        cls, conf, graft, reason = self._MOCK[i % len(self._MOCK)]
        return {"classification": cls, "confidence": conf,
                "graft_score": graft, "reason": reason}

    # ---- parse / validate --------------------------------------------------

    @staticmethod
    def _parse_array(text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if not m:
                raise AnalyzerError(f"could not parse JSON array: {text[:200]!r}")
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                raise AnalyzerError(f"could not parse JSON array: {text[:200]!r}")
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            raise AnalyzerError("model output is not a JSON array")
        return data

    @staticmethod
    def _coerce_float(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.0
        return round(max(0.0, min(1.0, f)), 3)

    @classmethod
    def _validate_one(cls, data):
        if not isinstance(data, dict):
            return {"classification": "empty", "confidence": 0.0,
                    "graft_score": 0.0, "reason": "no data"}
        classification = str(data.get("classification", "empty")).strip().lower()
        if classification not in CLASSES:
            classification = "empty"
        confidence = cls._coerce_float(data.get("confidence", 0.0))
        graft_score = cls._coerce_float(data.get("graft_score", 0.0))
        reason = str(data.get("reason", "")).strip() or "—"
        return {"classification": classification, "confidence": confidence,
                "graft_score": graft_score, "reason": reason}
