"""
opencv_filter.py — Stage 2: lightweight VISUAL reject filter (NO biology).

Its ONLY job is to cut Gemini's workload by dropping cells that are visually
unlikely to contain useful brood. It does NOT infer eggs / larvae / stage / any
biological meaning — that is entirely Gemini's job. It only flags cells that are
visually obvious non-targets:

  - reflection          : strong specular glare (near-white blob)
  - low_variance        : almost no intensity variation (flat patch)
  - uniform_texture     : low entropy + no edges (featureless)
  - sealed_appearance   : smooth, centre indistinguishable from the wax wall (capped)

CONSERVATIVE by design: keep-if-uncertain. False positives (passing a useless
cell) are fine; false negatives (skipping a real larva) must be minimized, so a
crop is skipped only on a strong, unambiguous cue. Texture thresholds are RELATIVE
to each image's own median (scale/contrast invariant); crops are normalized to a
fixed size first.

filter_cells(image, cells) -> [
    {"cell_id", "keep": bool, "reason": str|"", "metrics": {...}}
]
reason is "" when kept, else one of:
  reflection | low_variance | uniform_texture | sealed_appearance
"""

import cv2
import numpy as np


NORM = 64                  # normalize each crop to NORM x NORM (scale invariance)
REFLECT_LEVEL = 248        # pixels >= this count as specular glare
REFLECT_FRAC = 0.06        # skip if this fraction of the crop is glare
STD_REL = 0.45             # low_variance: std below STD_REL * median std
LAP_REL = 0.40             # "flat": lap_var below LAP_REL * median lap_var
ENT_REL = 0.62             # uniform: entropy below ENT_REL * median entropy
CS_REL = 0.55              # sealed: centre≈surround below CS_REL * median


def _metrics(gray_crop):
    g = cv2.resize(gray_crop, (NORM, NORM), interpolation=cv2.INTER_AREA)
    std = float(g.std())
    lap_var = float(cv2.Laplacian(g, cv2.CV_64F).var())

    hist = cv2.calcHist([g], [0], None, [32], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum()) if p.size else 0.0

    bright_frac = float((g >= REFLECT_LEVEL).mean())

    r = NORM // 4
    c0 = NORM // 2 - r
    center = g[c0:c0 + 2 * r, c0:c0 + 2 * r]
    cmean = float(center.mean())
    mask = np.ones_like(g, dtype=bool)
    mask[c0:c0 + 2 * r, c0:c0 + 2 * r] = False
    smean = float(g[mask].mean())

    return {
        "std": round(std, 2),
        "lap_var": round(lap_var, 2),
        "entropy": round(entropy, 3),
        "bright_frac": round(bright_frac, 3),
        "center_surround": round(abs(cmean - smean), 2),
    }


def _median(vals):
    return float(np.median(vals)) if len(vals) else 0.0


def filter_cells(image, cells):
    img = cv2.imread(image) if isinstance(image, str) else image
    if img is None:
        raise ValueError("opencv_filter: could not read image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    recs = []
    for c in cells:
        x1, y1, x2, y2 = c["bbox"]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(W, x2); y2 = min(H, y2)
        crop = gray[y1:y2, x1:x2]
        m = _metrics(crop) if crop.size >= 16 else None
        recs.append((c["cell_id"], m))

    ms = [m for _, m in recs if m]
    med_std = _median([m["std"] for m in ms]) or 1.0
    med_lap = _median([m["lap_var"] for m in ms]) or 1.0
    med_ent = _median([m["entropy"] for m in ms]) or 1.0
    med_cs = _median([m["center_surround"] for m in ms]) or 1.0

    out = []
    for cid, m in recs:
        if m is None:                                   # too small to judge -> keep
            out.append({"cell_id": cid, "keep": True, "reason": "", "metrics": {}})
            continue
        flat = m["lap_var"] < LAP_REL * med_lap         # supporting "no edges" cue

        if m["bright_frac"] >= REFLECT_FRAC:
            keep, reason = False, "reflection"
        elif m["std"] < STD_REL * med_std and flat:
            keep, reason = False, "low_variance"
        elif m["entropy"] < ENT_REL * med_ent and flat:
            keep, reason = False, "uniform_texture"
        elif m["center_surround"] < CS_REL * med_cs and flat:
            keep, reason = False, "sealed_appearance"
        else:
            keep, reason = True, ""
        out.append({"cell_id": cid, "keep": keep, "reason": reason, "metrics": m})
    return out
