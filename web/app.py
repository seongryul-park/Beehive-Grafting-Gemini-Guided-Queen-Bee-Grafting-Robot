"""
app.py — Flask server for the beehive grafting demo.

Architecture (see pipeline.py):
    image -> OpenCV geometric CELL detection -> one cell = one crop
          -> OpenCV fast visual filter -> Gemini biological classification
          -> ranking -> selection.
OpenCV does deterministic localization + cheap filtering; Gemini does the
biological reasoning, batched into ~32-48-crop calls and merged.

Endpoints:
  GET  /         -> dashboard (static/index.html)
  GET  /health   -> {status, model, mock, detector, confidence_threshold}
  POST /analyze  -> multipart "image"; full pipeline result (+ debug)
  POST /detect_opencv -> geometric honeycomb cell detection only (no Gemini)
"""

import os
import tempfile

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from cell_detector import CellDetector, DetectorError
from gemini_analyzer import GeminiAnalyzer, AnalyzerError
from pipeline import GraftingPipeline
import opencv_prototype

load_dotenv()


def _env(name, default=""):
    """Read an env var, tolerating an inline '# comment' that some python-dotenv
    versions leave attached to the value (which otherwise silently corrupts
    DETECTOR and bypasses the OpenCV stage)."""
    return os.getenv(name, default).split("#", 1)[0].strip()


GEMINI_API_KEY = _env("GEMINI_API_KEY", "")
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-2.5-flash")
FLASK_PORT = int(_env("FLASK_PORT", "5000") or "5000")
DETECTOR = _env("DETECTOR", "geometry").lower()            # geometry | opencv | color | gemini
CONFIDENCE_THRESHOLD = float(_env("CONFIDENCE_THRESHOLD", "0.7") or "0.7")
MAX_CELLS = int(_env("MAX_CELLS", "150") or "150")
GEMINI_BATCH_SIZE = int(_env("GEMINI_BATCH_SIZE", "48") or "48")  # ~32-48 crops/call
# Folder of curated ideal-young-larva reference crops prepended to every batch.
_DEFAULT_REF_DIR = os.path.join(os.path.dirname(__file__), "sample_images", "rag")
GEMINI_REFERENCE_DIR = _env("GEMINI_REFERENCE_DIR", _DEFAULT_REF_DIR) or _DEFAULT_REF_DIR
GEMINI_MOCK = _env("GEMINI_MOCK", "0").lower() in ("1", "true", "yes", "on")
DEBUG_DIR = _env("DEBUG_DIR", "debug")

if DETECTOR not in ("color", "opencv", "geometry", "gemini"):
    raise SystemExit(
        f"DETECTOR must be color|opencv|geometry|gemini, got {DETECTOR!r} "
        "(check for stray characters/comments in .env)"
    )

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Build one shared Gemini client (unless mocking). Missing key is not fatal:
# the server still boots so /health works and /analyze reports the problem.
client = None
client_error = None
if not GEMINI_MOCK:
    if not GEMINI_API_KEY:
        client_error = "GEMINI_API_KEY is not set"
    else:
        try:
            from google import genai
            client = genai.Client(api_key=GEMINI_API_KEY)
        except Exception as e:  # bad SDK / key
            client_error = str(e)

detector = CellDetector(
    method=DETECTOR, client=client, model=GEMINI_MODEL,
    mock=GEMINI_MOCK, max_cells=MAX_CELLS,
)
analyzer = GeminiAnalyzer(
    client=client, model=GEMINI_MODEL, mock=GEMINI_MOCK, max_cells=MAX_CELLS,
    batch_size=GEMINI_BATCH_SIZE, reference_dir=GEMINI_REFERENCE_DIR,
)
pipeline = GraftingPipeline(
    detector, analyzer, mock=GEMINI_MOCK,
    confidence_threshold=CONFIDENCE_THRESHOLD, debug_dir=DEBUG_DIR,
)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "mock" if GEMINI_MOCK else GEMINI_MODEL,
        "mock": GEMINI_MOCK,
        "detector": "mock" if GEMINI_MOCK else DETECTOR,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })


@app.post("/analyze")
def analyze():
    if not GEMINI_MOCK and client is None:
        return jsonify({"error": f"server misconfigured: {client_error}"}), 500
    return _with_upload(lambda path: pipeline.run(path), (DetectorError, AnalyzerError))


@app.post("/detect_opencv")
def detect_opencv():
    """OpenCV honeycomb CELL detection only (geometry) — no Gemini."""
    return _with_upload(lambda path: opencv_prototype.detect(path), (ValueError, Exception))


def _with_upload(fn, errors):
    if "image" not in request.files:
        return jsonify({"error": "no image uploaded (field name must be 'image')"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "empty filename"}), 400

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()
        return jsonify(fn(tmp.name))
    except Exception as e:                      # log the REAL cause to the console
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 502
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT)
