"""
vision_core.py — ROS-free wrapper around the EXISTING beehive vision pipeline.

This module is deliberately free of any ROS dependency so it can be unit-tested
on its own and reused. It imports the unchanged pipeline from web/ (pipeline.py,
cell_detector.py, gemini_analyzer.py, opencv_filter.py) exactly as web/app.py
wires it together, and exposes three things:

    VisionPipeline(...)              build the pipeline once (mirrors app.py)
    VisionPipeline.analyze_bgr(img)  run it on an in-memory BGR frame -> result
    annotate(bgr, result)            draw cells / candidates / selected cell

The pipeline is NOT modified: analyze_bgr writes the frame to a temp PNG and
calls pipeline.run(path), because run() takes an image path. The robotics layer
(vision_node) consumes this; the pipeline stays a black box.
"""

import os
import sys
import tempfile


def add_web_to_path(web_dir: str) -> str:
    """Put the beehive web/ folder on sys.path so its modules import cleanly."""
    web_dir = os.path.abspath(os.path.expanduser(web_dir))
    if not os.path.isdir(web_dir):
        raise FileNotFoundError(f"web_dir does not exist: {web_dir}")
    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)
    return web_dir


class VisionPipeline:
    """Builds the existing GraftingPipeline once and runs it on raw frames.

    Parameters mirror web/app.py's env-driven construction so behaviour matches
    the standalone Flask server. Defaults to mock=True so the node runs with no
    API key (real OpenCV detection still happens; only Gemini is mocked).
    """

    def __init__(self, web_dir, *, mock=True, detector="color",
                 model="gemini-2.5-flash", api_key="", confidence_threshold=0.7,
                 max_cells=150, batch_size=48, reference_dir=None,
                 debug_dir="debug"):
        add_web_to_path(web_dir)
        # Imported here (not at module top) so vision_core has no hard dep on the
        # web/ code until a pipeline is actually constructed.
        from cell_detector import CellDetector
        from gemini_analyzer import GeminiAnalyzer, REF_DIR_DEFAULT
        from pipeline import GraftingPipeline

        client = None
        if not mock and api_key:
            from google import genai
            client = genai.Client(api_key=api_key)

        if reference_dir is None:
            reference_dir = REF_DIR_DEFAULT

        detector_obj = CellDetector(
            method=detector, client=client, model=model,
            mock=mock, max_cells=max_cells,
        )
        analyzer_obj = GeminiAnalyzer(
            client=client, model=model, mock=mock, max_cells=max_cells,
            batch_size=batch_size, reference_dir=reference_dir,
        )
        self._pipeline = GraftingPipeline(
            detector_obj, analyzer_obj, mock=mock,
            confidence_threshold=confidence_threshold, debug_dir=debug_dir,
        )
        self.mock = mock
        self.detector = detector

    def analyze_bgr(self, bgr):
        """Run the existing pipeline on an in-memory BGR image. Returns its dict."""
        import cv2
        fd, path = tempfile.mkstemp(prefix="beehive_frame_", suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(path, bgr):
                raise RuntimeError("failed to write frame to temp file")
            return self._pipeline.run(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def selected_cell(result):
    """Return the selected cell record (dict) from a pipeline result, or None."""
    if not result:
        return None
    sid = result.get("selected")
    if sid is None:
        return None
    for cell in result.get("cells", []):
        if cell.get("cell_id") == sid:
            return cell
    return None


def selected_pixel(result):
    """Return (px, py) image-pixel center of the selected cell, or None.

    This is the hand-off point to the future Stage-4 coordinate transform; it is
    intentionally just a pixel here — no 3D mapping is done in Stage 1.
    """
    cell = selected_cell(result)
    if cell is None:
        return None
    if "center" in cell and cell["center"]:
        cx, cy = cell["center"][0], cell["center"][1]
        return int(cx), int(cy)
    x1, y1, x2, y2 = cell["bbox"]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def annotate(bgr, result,
             selected_color=(0, 230, 0),
             candidate_color=(0, 200, 255),
             cell_color=(110, 110, 110)):
    """Draw the pipeline result onto a copy of the BGR frame for live viewing.

    - every detected cell: thin grey box
    - ranked candidates:   amber box
    - selected cell:       thick green box + label + crosshair at its center
    A small header shows counts and the selection. Safe if result is None/empty.
    """
    import cv2
    img = bgr.copy()
    cells = (result or {}).get("cells", [])
    candidate_ids = {c["cell_id"] for c in (result or {}).get("candidates", [])}
    selected = (result or {}).get("selected")

    for cell in cells:
        x1, y1, x2, y2 = cell["bbox"]
        cid = cell.get("cell_id")
        if cid == selected:
            color, thick = selected_color, 3
        elif cid in candidate_ids:
            color, thick = candidate_color, 2
        else:
            color, thick = cell_color, 1
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)

    px = selected_pixel(result)
    if px is not None:
        cx, cy = px
        cv2.drawMarker(img, (cx, cy), selected_color, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(img, f"#{selected}", (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, selected_color, 2)

    sel_txt = f"selected #{selected}" if selected is not None else "no selection"
    header = f"cells:{len(cells)}  candidates:{len(candidate_ids)}  {sel_txt}"
    cv2.putText(img, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, header, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return img
