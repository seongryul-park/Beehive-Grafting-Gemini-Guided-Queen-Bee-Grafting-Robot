"""
pipeline.py — full vision pipeline.

  image
   -> Stage 1: OpenCV geometric CELL detection   (cell_detector / opencv_prototype)
   -> one cell = one crop
   -> Stage 2: OpenCV lightweight visual FILTER   (opencv_filter) — reject-only, NO biology
   -> Sort surviving crops by OpenCV visual priority (highest first)
   -> Stage 3: Gemini biological CLASSIFICATION   (gemini_analyzer) — BATCHED
        candidates are split into batches of ~32-48 crops; each batch is one
        Gemini call; all batch outputs are merged by cell_id
   -> Ranking (graft_score desc, confidence desc) — only AFTER every batch is merged
   -> Selection

OpenCV = deterministic localization + cheap visual rejection.
Gemini = ALL biological reasoning / graft decision.
The filter shrinks how many crops Gemini sees; batching keeps each request small
enough to stay reliable. A single failed batch is reported but does not lose the
other batches' results.

Every run prints an expanded summary + per-cell table and writes a debug/ folder
(images, gemini_results.json, pipeline_summary.txt, pipeline_results.csv).
"""

import csv
import json
import os
import time

import opencv_filter


_CLASS_LABEL = {"egg": "Egg", "young_larva": "Young Larva",
                "old_larva": "Old Larva", "empty": "Empty"}
# filter reason tag -> human label for the summary "Reason Counts" block
_REASON_LABEL = [
    ("uniform_texture", "Uniform texture"),
    ("low_variance", "Low variance"),
    ("reflection", "Reflection"),
    ("sealed_appearance", "Sealed appearance"),
]


# OpenCV visual-priority cues: more texture / edges / centre-vs-surround
# contrast => more likely to hold brood => analyse first.
_PRIORITY_KEYS = ("lap_var", "std", "entropy", "center_surround")


def _visual_priority(kept, fmap):
    """Order kept cells highest-to-lowest by an OpenCV visual-priority score.

    Score = sum of each filter metric normalized by that metric's max across the
    kept set, so all four cues contribute comparably regardless of scale. Cells
    with no metrics (too small to measure) score 0 and sort last. Stable on
    cell_id for deterministic batching.
    """
    metrics = {c["cell_id"]: (fmap[c["cell_id"]].get("metrics") or {}) for c in kept}
    maxes = {
        k: max((m.get(k, 0.0) for m in metrics.values()), default=0.0) or 1.0
        for k in _PRIORITY_KEYS
    }

    def score(c):
        m = metrics[c["cell_id"]]
        return sum(m.get(k, 0.0) / maxes[k] for k in _PRIORITY_KEYS)

    return sorted(kept, key=lambda c: (-score(c), c["cell_id"]))


class GraftingPipeline:
    def __init__(self, detector, analyzer, mock=False,
                 confidence_threshold=0.7, debug_dir="debug"):
        self.detector = detector
        self.analyzer = analyzer
        self.mock = mock
        self.confidence_threshold = confidence_threshold
        self.debug_dir = debug_dir

    def run(self, image_path: str) -> dict:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("pipeline: could not read image")
        H, W = img.shape[:2]

        # Stage 1: detect honeycomb cells.
        det = self.detector.detect(image_path)
        cells = det["cells"]                      # [{cell_id, bbox}]

        # Stage 2: lightweight visual filter (no biology).
        filt = opencv_filter.filter_cells(img, cells)
        fmap = {f["cell_id"]: f for f in filt}
        kept = [c for c in cells if fmap[c["cell_id"]]["keep"]]

        reason_counts = {tag: 0 for tag, _ in _REASON_LABEL}
        for f in filt:
            if not f["keep"]:
                reason_counts[f["reason"]] = reason_counts.get(f["reason"], 0) + 1

        # Sort surviving crops by OpenCV visual priority (highest first) so the
        # most promising cells lead the batches.
        ordered = _visual_priority(kept, fmap)

        # Stage 3: BATCHED Gemini classification of the ordered crops. The
        # analyzer splits them into ~32-48-crop batches, one call each, and
        # merges every batch's output by cell_id before returning.
        verdicts = {}
        gemini_calls = 0
        batch_info = []
        failed_batches = []
        if ordered:
            res = self.analyzer.classify_cells(image_path, ordered)
            verdicts = {v["cell_id"]: v for v in res["cells"]}
            gemini_calls = res.get("gemini_calls", 0)
            batch_info = res.get("batches", [])
            failed_batches = res.get("failed_batches", [])

        # Ranking happens ONLY after all batches are merged.
        # graft_score desc, then confidence desc. Empty/low-conf excluded.
        survivors = [
            v for v in verdicts.values()
            if v["classification"] != "empty"
            and v["confidence"] >= self.confidence_threshold
        ]
        ranked = sorted(
            survivors,
            key=lambda v: (v.get("graft_score", 0.0), v.get("confidence", 0.0)),
            reverse=True,
        )
        selected = ranked[0]["cell_id"] if ranked else None

        # Unified per-cell records (detection order) for response + debug.
        unified = []
        for c in cells:
            cid = c["cell_id"]
            x1, y1, x2, y2 = c["bbox"]
            f = fmap[cid]
            row = {
                "cell_id": cid,
                "bbox": c["bbox"],
                "center": [(x1 + x2) // 2, (y1 + y2) // 2],
                "filter": "PASS" if f["keep"] else "SKIP",
                "filter_reason": f["reason"],
            }
            if cid in verdicts:
                v = verdicts[cid]
                row.update({
                    "classification": v["classification"],
                    "confidence": v["confidence"],
                    "graft_score": v["graft_score"],
                    "reason": v["reason"],
                })
            row["decision"] = (
                "Selected" if cid == selected
                else ("Reject" if cid in verdicts else f["reason"])
            )
            unified.append(row)

        counts = {k: 0 for k in _CLASS_LABEL}
        for v in verdicts.values():
            counts[v["classification"]] = counts.get(v["classification"], 0) + 1

        summary = self._summary(W, H, len(cells), len(cells) - len(kept),
                                reason_counts, len(kept), gemini_calls, counts,
                                selected, verdicts.get(selected),
                                batch_info, len(verdicts), failed_batches)
        table = self._table(unified)
        print(summary)
        print(table)

        debug_dir = self._write_debug(img, cells, fmap, kept, verdicts, unified,
                                      summary, table, selected)

        candidates = [
            {"cell_id": v["cell_id"], "classification": v["classification"],
             "confidence": v["confidence"], "graft_score": v["graft_score"]}
            for v in ranked
        ]
        return {
            "image": {"width": W, "height": H},
            "cells": unified,
            "candidates": candidates,
            "selected": selected,
            "detector": "mock" if self.mock else self.detector.method,
            "gemini_calls": gemini_calls,
            "confidence_threshold": self.confidence_threshold,
            "stages": {
                "geometry_cells": len(cells),
                "filtered_out": len(cells) - len(kept),
                "filter_reasons": reason_counts,
                "sent_to_gemini": len(kept),
                "batch_count": len(batch_info),
                "batches": batch_info,
                "merged_results": len(verdicts),
                "failed_batches": failed_batches,
            },
            "gemini_results": counts,
            "debug_dir": debug_dir,
        }

    # ---- debug text --------------------------------------------------------

    @staticmethod
    def _summary(W, H, n_cells, n_removed, reason_counts, n_sent, calls,
                 counts, selected, sel_v, batch_info=None, merged=0,
                 failed_batches=None):
        batch_info = batch_info or []
        failed_batches = failed_batches or []
        sl = "—"; sconf = "—"; sgr = "—"
        if selected is not None and sel_v:
            sl = f"#{selected} ({_CLASS_LABEL.get(sel_v['classification'], sel_v['classification'])})"
            sconf = f"{sel_v['confidence']:.2f}"
            sgr = f"{sel_v['graft_score']:.2f}"
        n_filtered = n_cells - n_removed
        lines = [
            "\n=================================================",
            f"Image Size               : {W} x {H}",
            f"Geometry Cells           : {n_cells}",
            f"Cells Removed by OpenCV  : {n_removed}",
            "Reason Counts",
        ]
        for tag, label in _REASON_LABEL:
            lines.append(f"  {label:<22}: {reason_counts.get(tag, 0)}")
        lines += [
            f"Filtered Cells           : {n_filtered}",
            f"Cells Sent to Gemini     : {n_sent}",
            f"Batch Count              : {len(batch_info)}",
        ]
        for b in batch_info:
            status = "" if b.get("ok", True) else "  [FAILED]"
            lines.append(f"  Batch {b['index']:<16}: {b['size']} cells{status}")
        lines += [
            f"Gemini Calls             : {calls}",
            f"Merged Results           : {merged}",
            f"Batch Failures           : {len(failed_batches)}",
        ]
        for b in failed_batches:
            lines.append(f"  Batch {b['index']} failed       : {b['error']}")
        lines += [
            "Gemini Classification Counts",
            f"  Egg                   : {counts.get('egg', 0)}",
            f"  Young Larva           : {counts.get('young_larva', 0)}",
            f"  Old Larva             : {counts.get('old_larva', 0)}",
            f"  Empty                 : {counts.get('empty', 0)}",
            f"Final Selected Cell      : {sl}",
            f"Confidence               : {sconf}",
            f"Graft Score              : {sgr}",
            "=================================================\n",
        ]
        return "\n".join(lines)

    @staticmethod
    def _table(unified):
        lines = ["ID  | Filter | Reason            | Classification | Conf | Graft | Decision",
                 "----+--------+-------------------+----------------+------+-------+----------"]
        for r in unified:
            cls = r.get("classification", "--")
            conf = f"{r['confidence']:.2f}" if "confidence" in r else "--"
            gr = f"{r['graft_score']:.2f}" if "graft_score" in r else "--"
            lines.append(
                f"{r['cell_id']:<3} | {r['filter']:<6} | {(r['filter_reason'] or ''):<17} | "
                f"{cls:<14} | {conf:<4} | {gr:<5} | {r['decision']}"
            )
        return "\n".join(lines)

    # ---- debug images / files ---------------------------------------------

    def _write_debug(self, img, cells, fmap, kept, verdicts, unified,
                     summary, table, selected):
        try:
            import cv2
            run = os.path.join(self.debug_dir, time.strftime("run_%Y%m%d_%H%M%S"))
            os.makedirs(run, exist_ok=True)

            geo = img.copy()
            for c in cells:
                x1, y1, x2, y2 = c["bbox"]
                cv2.rectangle(geo, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.imwrite(os.path.join(run, "geometry_detection.png"), geo)

            filt_img = img.copy()
            for c in cells:
                x1, y1, x2, y2 = c["bbox"]
                keep = fmap[c["cell_id"]]["keep"]
                cv2.rectangle(filt_img, (x1, y1), (x2, y2),
                              (0, 200, 0) if keep else (0, 0, 230), 2)
            cv2.imwrite(os.path.join(run, "opencv_filter.png"), filt_img)

            # Reference-gallery crops: one clean single-cell image per DETECTED
            # cell, written to debug/<run>/crops/crop_<cell_id>.png with the
            # cell_id preserved in the filename, plus an unlabeled crops.csv
            # (cell_id, crop_filename, center_x, center_y, radius) for manual
            # review. No classification is written here on purpose.
            crops_dir = os.path.join(run, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            crop_rows = []
            for c in cells:
                x1, y1, x2, y2 = c["bbox"]
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if not crop.size:
                    continue
                fname = f"crop_{c['cell_id']:03d}.png"
                cv2.imwrite(os.path.join(crops_dir, fname), crop)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                radius = round(((x2 - x1) + (y2 - y1)) / 4)
                crop_rows.append([c["cell_id"], fname, cx, cy, radius])

            with open(os.path.join(crops_dir, "crops.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["cell_id", "crop_filename", "center_x", "center_y", "radius"])
                w.writerows(crop_rows)

            with open(os.path.join(run, "gemini_results.json"), "w") as fh:
                json.dump({"selected": selected, "cells": list(verdicts.values())},
                          fh, indent=2)
            with open(os.path.join(run, "pipeline_summary.txt"), "w") as fh:
                fh.write(summary + "\n" + table + "\n")

            # CSV: cell_id,opencv_filter,filter_reason,classification,confidence,graft_score
            with open(os.path.join(run, "pipeline_results.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["cell_id", "opencv_filter", "filter_reason",
                            "classification", "confidence", "graft_score"])
                for r in unified:
                    w.writerow([
                        r["cell_id"], r["filter"], r["filter_reason"],
                        r.get("classification", ""),
                        r.get("confidence", ""), r.get("graft_score", ""),
                    ])
            return run
        except Exception as e:  # debug must never break the response
            print(f"[debug] could not write debug dir: {e}")
            return None
