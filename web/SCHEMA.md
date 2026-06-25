# Gemini Response Schema — Single Source of Truth

## Architecture

```
image → OpenCV candidate generation → Gemini biological verification
      → classification → filtering → ranking → selection
```

- **OpenCV (`color_segment.py`)** is a *recall-oriented* candidate proposer. It
  intentionally over-proposes — reflections, bright wax, empty cells, eggs, and
  other artifacts are expected and acceptable. Missing a real larva is worse than
  a few false positives. It does **not** classify.
- **Gemini (`gemini_analyzer.py`)** is the reasoning/decision engine. For **every**
  candidate it performs biological verification, stage estimation, egg detection,
  and graft-suitability scoring, in **one call per image**.
- **`pipeline.py`** does pure, deterministic post-processing: filter → rank →
  select. No extra model call.

---

## 1. Per-candidate schema from the single Gemini call

The model returns a **JSON array**, one object per candidate:

```jsonc
[
  {
    "cell_id": 1,                     // int, matches the OpenCV candidate id
    "exists": true,                   // bool — biologically meaningful brood present?
    "classification": "young_larva",  // egg | young_larva | old_larva | empty | unknown
    "confidence": 0.80,               // float 0..1 — certainty in the classification
    "graft_score": 0.90,              // float 0..1 — graft suitability, INDEPENDENT of confidence
    "reason": "small C-shaped larva in royal jelly"
  }
]
```

(When `DETECTOR=gemini`, the same object additionally carries `bbox`, since Gemini
also detects. With `DETECTOR=color`/`opencv`, the server supplies `bbox` from the
detector candidate.)

Field contract:

| field            | type        | constraint / meaning                                                |
|------------------|-------------|---------------------------------------------------------------------|
| `cell_id`        | int         | stable join key, contiguous `1..N`                                  |
| `exists`         | bool        | `true` iff real brood (egg/larva); `false` for reflection/wax/empty/artifact |
| `classification` | string enum | `egg`, `young_larva`, `old_larva`, `empty`, `unknown`               |
| `confidence`     | float       | `0..1` — certainty in the classification                            |
| `graft_score`    | float       | `0..1` — suitability for queen grafting, **independent** of confidence |
| `reason`         | string      | short justification                                                 |

`confidence` and `graft_score` are deliberately decoupled:

| example                                   | confidence | graft_score |
|-------------------------------------------|-----------|-------------|
| clearly an old larva (sure, but too old)  | 0.95      | 0.15        |
| probably a young larva (ideal if right)   | 0.80      | 0.90        |

### Server normalization

`gemini_analyzer._validate_one` enforces the contract before anything downstream
sees it:

- `classification` not in the enum → `"unknown"`.
- `confidence` / `graft_score` missing or non-numeric → `0.0`; both clamped to `[0,1]`.
- `exists` trusts an explicit bool/`"true"`; otherwise inferred from the class.
  A non-brood class (`empty`/`unknown`) is **always forced** to `exists=false`.
- `reason` empty → `"—"`.
- `cell_id` preserved from the candidate (contiguous `1..N`).

---

## 2. Filtering → Ranking → Selection (`pipeline.py`)

```python
# FILTER — drop a candidate if it is not real brood, or the verdict is shaky
survivors = [c for c in cells
             if c["exists"] and c["confidence"] >= CONFIDENCE_THRESHOLD]

# RANK — graft_score first, confidence as the tie-breaker
ranked = sorted(survivors,
                key=lambda c: (c["graft_score"], c["confidence"]),
                reverse=True)

# SELECT — best-ranked survivor
selected = ranked[0]["cell_id"] if ranked else None
```

Rules:
1. **Discard** when `exists == false` **OR** `confidence < CONFIDENCE_THRESHOLD`
   (env `CONFIDENCE_THRESHOLD`, default `0.7`).
2. **Rank** survivors by `graft_score` desc, then `confidence` desc. Note ranking
   is driven by graft suitability, **not** by class — a high-confidence `old_larva`
   ranks below a young larva because its `graft_score` is low.
3. **Select** the top-ranked survivor's `cell_id`, or `null` if none survive.

---

## 3. Final response from `POST /analyze`

```json
{
  "image": { "width": 360, "height": 270 },
  "cells": [
    { "cell_id": 1, "bbox": [.. ], "exists": false, "classification": "empty",       "confidence": 0.95, "graft_score": 0.00, "reason": "clean drawn comb" },
    { "cell_id": 3, "bbox": [.. ], "exists": true,  "classification": "young_larva", "confidence": 0.94, "graft_score": 0.95, "reason": "C-shaped larva in royal jelly" },
    { "cell_id": 4, "bbox": [.. ], "exists": true,  "classification": "old_larva",   "confidence": 0.80, "graft_score": 0.15, "reason": "plump larva, past grafting age" }
  ],
  "candidates": [
    { "cell_id": 3, "classification": "young_larva", "confidence": 0.94, "graft_score": 0.95 }
  ],
  "selected": 3,
  "detector": "color",
  "gemini_calls": 1,
  "confidence_threshold": 0.7,
  "verification": {
    "candidates": 12,
    "rejected_not_exists": 4,
    "rejected_low_confidence": 1,
    "survivors": 7
  }
}
```

Top-level fields:

| field                  | type        | meaning                                                        |
|------------------------|-------------|----------------------------------------------------------------|
| `image`                | object      | `{width, height}` in pixels                                    |
| `cells`                | object[]    | every OpenCV candidate with Gemini's full verdict (§1)         |
| `candidates`           | object[]    | **survivors, ranked** (graft_score desc, confidence desc)      |
| `selected`             | int \| null | `cell_id` of the chosen graft target                           |
| `detector`             | string      | `color`, `opencv`, `gemini`, or `mock`                         |
| `gemini_calls`         | int         | API calls this request (`0` mock, `1` live)                    |
| `confidence_threshold` | float       | filter cutoff applied this run                                 |
| `verification`         | object      | candidate funnel counts (proposed → rejected → survivors)      |

Error case (any stage fails): HTTP `4xx/5xx` with `{ "error": "<message>" }`.

---

## 4. Consumer contract

- **Dashboard overlay** — `cells[].bbox` rectangle, color by `classification`,
  dim/strike candidates with `exists=false`, label with `confidence` + `graft_score`,
  `selected` gets the target halo; `verification` drives the funnel telemetry.
- **Ranking panel** — `candidates[]` is already filtered + sorted; render as-is.
  `selected` gets the `GRAFT` badge.
- **ROS publisher** — `selected` first, then `candidates[]` order as the work queue;
  per target publish the bbox pixel center `((x1+x2)/2,(y1+y2)/2)`.
- **MoveIt** — consumes the pixel anchor, maps pixel→robot frame via the cell→coord
  lookup (calibration lives on the robot side).

Invariants: `cells` non-empty on success; every cell has all six fields,
server-normalized; `confidence, graft_score ∈ [0,1]`; `candidates` are all
`exists=true` and `confidence ≥ threshold`, sorted; `selected = candidates[0].cell_id`
or `null`; `cell_id` is the stable key across overlay ↔ ranking ↔ ROS ↔ MoveIt.
