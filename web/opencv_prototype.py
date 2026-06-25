"""
opencv_prototype.py — honeycomb CELL detector (geometry, NO Gemini).

Stage 1: detect the biological honeycomb cells from the wax-wall geometry (NOT
larvae, NOT bright regions). Each detected cell becomes ONE single-cell crop for
Gemini; boxes never span multiple cells.

  1. grayscale + CLAHE
  2. candidate centers: HoughCircles (cell openings) UNION adaptive-threshold
     contour centroids (closed wax regions)
  3. estimate median cell radius R; clamp candidate radii to a band around R
  4. aggressive center-distance de-dup -> one center per cell
  5. UNIFORM single-cell box sized to the global median radius (+ small margin)

detect(image_path) -> {
    "width","height","cell_radius","avg_cell_size",
    "cells":[{"cell_id","center":[cx,cy],"radius":r,"bbox":[x1,y1,x2,y2]}, ...],
    "debug":{raw_candidates,hough_count,contour_count,duplicate_removals,
             detected_centers,cell_radius,avg_cell_size}
}
"""

import json
import math
import sys

import cv2
import numpy as np

DEDUP_FACTOR = 1.3
RADIUS_LO = 0.6
RADIUS_HI = 1.7
MARGIN_FRAC = 0.10


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _candidates(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"could not read image: {image_path}")
    H, W = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    min_r = max(6, int(min(W, H) / 45))
    max_r = max(min_r + 6, int(min(W, H) / 10))

    cands = []
    circles = cv2.HoughCircles(
        eq, cv2.HOUGH_GRADIENT, dp=1.0, minDist=int(min_r * 1.8),
        param1=90, param2=26, minRadius=min_r, maxRadius=max_r,
    )
    hough_count = 0
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        hough_count = len(circles)
        for x, y, r in circles:
            cands.append((float(x), float(y), float(r)))

    th = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=31, C=5,
    )
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (min_r * min_r) * 1.2
    max_area = (max_r * 2.0) ** 2
    contour_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h else 0
        if ar < 0.5 or ar > 2.0:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        r = math.sqrt(area / math.pi)
        cands.append((cx, cy, r))
        contour_count += 1

    return W, H, cands, hough_count, contour_count


def _dedupe_centers(cands, merge_dist):
    merge_sq = merge_dist * merge_dist
    ordered = sorted(cands, key=lambda c: c[2], reverse=True)
    kept = []
    for cx, cy, r in ordered:
        if all((cx - kx) ** 2 + (cy - ky) ** 2 >= merge_sq for kx, ky, _ in kept):
            kept.append((cx, cy, r))
    return kept, len(cands) - len(kept)


def detect(image_path, dedup_factor=DEDUP_FACTOR, margin_frac=MARGIN_FRAC,
           max_cells=4000):
    W, H, cands, hough_count, contour_count = _candidates(image_path)
    raw = len(cands)
    if not cands:
        raise ValueError("opencv found no honeycomb cells (try DETECTOR=color/gemini)")

    R = _median([r for _, _, r in cands]) or 1.0
    R = max(4.0, R)
    lo, hi = RADIUS_LO * R, RADIUS_HI * R
    cands = [(x, y, min(max(r, lo), hi)) for x, y, r in cands]

    kept, removals = _dedupe_centers(cands, merge_dist=dedup_factor * R)
    kept = kept[:max_cells]

    # Uniform single-cell box sized to the GLOBAL median cell radius (+ margin).
    half = max(3, int(round(R * (1.0 + margin_frac))))
    cells = []
    for cx, cy, r in kept:
        cxi, cyi, ri = int(round(cx)), int(round(cy)), int(round(r))
        x1 = max(0, cxi - half); y1 = max(0, cyi - half)
        x2 = min(W, cxi + half); y2 = min(H, cyi + half)
        if (x2 - x1) < 4 or (y2 - y1) < 4:
            continue
        cells.append({
            "cell_id": len(cells) + 1,
            "center": [cxi, cyi],
            "radius": ri,
            "bbox": [x1, y1, x2, y2],
        })

    if not cells:
        raise ValueError("opencv produced no valid single-cell boxes")

    avg_cell_size = round(2.0 * R, 2)
    return {
        "width": W, "height": H,
        "cell_radius": round(R, 2),
        "avg_cell_size": avg_cell_size,
        "cells": cells,
        "debug": {
            "raw_candidates": raw,
            "hough_count": hough_count,
            "contour_count": contour_count,
            "duplicate_removals": removals,
            "detected_centers": len(cells),
            "cell_radius": round(R, 2),
            "avg_cell_size": avg_cell_size,
        },
    }


def annotate(image_path, result, out_path):
    img = cv2.imread(image_path)
    for c in result["cells"]:
        x1, y1, x2, y2 = c["bbox"]
        cx, cy = c["center"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), -1)
    d = result["debug"]
    label = f"cells: {d['detected_centers']}  cell_radius: {d['cell_radius']:.0f}px  dups removed: {d['duplicate_removals']}"
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python opencv_prototype.py input.jpg [output.jpg]")
        sys.exit(1)
    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) > 2 else "cells_detected.jpg"
    res = detect(inp)
    print(json.dumps({"detected_centers": res["debug"]["detected_centers"],
                      "cell_radius": res["cell_radius"], "debug": res["debug"]}, indent=2))
    annotate(inp, res, outp)
    print(f"\nwrote -> {outp}")
