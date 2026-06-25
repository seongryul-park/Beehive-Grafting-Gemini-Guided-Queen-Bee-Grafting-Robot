"""
color_segment.py — per-cell larva candidate generator by COLOR/INTENSITY (NO Gemini).

Goal: propose ONE candidate per honeycomb cell that contains pale brood, so each
candidate crop sent to Gemini contains EXACTLY ONE cell (never a merged strip).

Approach:
  1. bright-and-pale brood mask (HSV ∩ LAB)               -> WHERE brood is
  2. estimate the honeycomb CELL radius from comb geometry -> HOW big a cell is
     (HoughCircles on the cell openings — NOT the thin larva blob, which would
      under-size the crop and over-segment)
  3. distance-transform the brood mask, take local maxima  -> per-cell CENTERS
     spaced >= ~1.1 cell-radius apart (this splits touching larvae)
  4. emit a single-cell SQUARE box (cell radius + small margin) around each center

detect(image_path) -> {
    "width": W, "height": H, "method": "color",
    "avg_cell_size": <2*R>, "cell_radius": <R>,
    "cells": [{"cell_id", "center":[cx,cy], "bbox":[x1,y1,x2,y2], "brightness"}, ...],
    "debug": {raw_peaks, kept_components, duplicate_removals,
              mask_fill_pct, v_thresh, s_cap, cell_radius, avg_cell_size}
}

CLI:
    python color_segment.py input.jpg [output.jpg]
"""

import json
import sys

import cv2
import numpy as np


# Saturation cap: larvae are pale, so keep only low-saturation bright pixels.
S_CAP = 110
# LAB b-channel neutral band: yellow comb has b >> 128; white larva ~128.
B_NEUTRAL = 165
# Single-cell crop margin, as a fraction of the estimated cell radius.
MARGIN_FRAC = 0.15


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _larva_mask(img, s_cap=S_CAP, b_neutral=B_NEUTRAL):
    """Bright-and-pale brood mask from HSV ∩ LAB. Returns (mask, v_thresh)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    _Hh, Ss, Vv = cv2.split(hsv)
    Ll, _aa, Bb = cv2.split(lab)

    Vv = cv2.GaussianBlur(Vv, (5, 5), 0)
    v_otsu, _ = cv2.threshold(Vv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thresh = int(max(v_otsu, 0.55 * 255))

    bright = (Vv >= v_thresh)
    pale = (Ss <= s_cap)
    l_bright = (Ll >= v_thresh)
    not_yellow = (Bb <= b_neutral)

    mask = (bright & pale & l_bright & not_yellow).astype(np.uint8) * 255
    return mask, v_thresh


def _estimate_cell_radius(img, W, H):
    """Estimate the honeycomb CELL radius from comb geometry (not larva size).

    The larva can be a thin C-shape, so its blob thickness is a poor proxy for how
    big a crop must be to hold one whole cell. HoughCircles on the cell openings
    gives the real cell radius. Falls back to an image-scale guess if too few.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    min_r = max(6, int(min(W, H) / 45))
    max_r = max(min_r + 6, int(min(W, H) / 10))
    circles = cv2.HoughCircles(
        eq, cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(min_r * 1.8),
        param1=90, param2=26, minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None and len(circles[0]) >= 3:
        return float(_median([float(c[2]) for c in circles[0]]))
    return float(max(8, min(W, H) / 22))


def _peak_centers(dist, sep, floor):
    """Local maxima of the distance transform, >= floor, separated by ~sep px.
    Returns list of (x, y, value) sorted by value descending."""
    ksize = max(3, int(sep) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dil = cv2.dilate(dist, kernel)
    peak = (dist == dil) & (dist >= floor)
    ys, xs = np.where(peak)
    pts = list(zip(xs.tolist(), ys.tolist(), dist[ys, xs].tolist()))
    pts.sort(key=lambda p: -p[2])
    return pts


def detect(image_path, s_cap=S_CAP, b_neutral=B_NEUTRAL,
           margin_frac=MARGIN_FRAC, max_cells=2000):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    H, W = img.shape[:2]

    mask, v_thresh = _larva_mask(img, s_cap, b_neutral)
    mask_fill_pct = round(100.0 * float(mask.mean()) / 255.0, 2)

    # Light clean only. A heavy CLOSE would bridge neighbouring cells across the
    # wax walls and re-merge them — exactly what we are trying to avoid.
    k3 = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k3, iterations=1)
    if mask.max() == 0:
        raise ValueError(
            "color segmentation found no brood-like regions "
            "(tune S_CAP / B_NEUTRAL or try the geometric detector)"
        )

    # Honeycomb CELL radius from comb geometry — sets crop size and the minimum
    # spacing between candidate centers (so we get ONE candidate per cell).
    R = _estimate_cell_radius(img, W, H)

    # Brood peaks: where pale brood sits. Spaced so each cell yields one center.
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peaks = _peak_centers(dist, sep=max(3, int(round(R * 0.8))), floor=2.0)
    if not peaks:
        raise ValueError("color segmentation found no brood centers")

    sep = max(3, int(round(R * 1.1)))
    sep_sq = sep * sep
    kept = []
    for x, y, v in peaks:
        if all((x - kx) ** 2 + (y - ky) ** 2 >= sep_sq for kx, ky, _ in kept):
            kept.append((x, y, v))
    raw_peaks = len(peaks)
    kept = kept[:max_cells]

    # Single-cell SQUARE box sized to the CELL radius (+ small margin).
    half = max(3, int(round(R * (1.0 + margin_frac))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cells = []
    for (x, y, _v) in kept:
        x1 = max(0, x - half); y1 = max(0, y - half)
        x2 = min(W, x + half); y2 = min(H, y + half)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        cells.append({
            "cell_id": len(cells) + 1,
            "center": [int(x), int(y)],
            "bbox": [x1, y1, x2, y2],
            "brightness": int(gray[int(y), int(x)]),
        })

    if not cells:
        raise ValueError("color segmentation produced no valid single-cell boxes")

    avg_cell_size = round(2.0 * R, 2)
    return {
        "width": W, "height": H,
        "method": "color",
        "avg_cell_size": avg_cell_size,
        "cell_radius": round(R, 2),
        "cells": cells,
        "debug": {
            "raw_peaks": raw_peaks,
            "kept_components": len(cells),
            "duplicate_removals": raw_peaks - len(cells),
            "mask_fill_pct": mask_fill_pct,
            "v_thresh": v_thresh,
            "s_cap": s_cap,
            "cell_radius": round(R, 2),
            "avg_cell_size": avg_cell_size,
        },
    }


def annotate(image_path, result, out_path):
    """Visualization: green mask tint + one single-cell square per candidate."""
    img = cv2.imread(image_path)
    mask, _ = _larva_mask(img)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    overlay = img.copy()
    overlay[mask > 0] = (0, 255, 0)
    img = cv2.addWeighted(overlay, 0.30, img, 0.70, 0)

    for c in result["cells"]:
        x1, y1, x2, y2 = c["bbox"]
        cx, cy = c["center"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    d = result["debug"]
    label = f"cells: {d['kept_components']}  cell_radius: {d['cell_radius']:.0f}px  merged-giants: 0"
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python color_segment.py input.jpg [output.jpg]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else "color_detected.jpg"
    res = detect(inp)
    print(json.dumps({"method": res["method"], "candidates": res["debug"]["kept_components"],
                      "cell_radius": res["cell_radius"], "debug": res["debug"]}, indent=2))
    annotate(inp, res, outp)
    print(f"\nwrote visualization -> {outp}")
