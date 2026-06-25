# 🐝 Beehive Grafting Robot

A one-day hackathon project (Google **Gemini API** theme, agriculture domain): a
vision system finds the larva best suited for queen **grafting (이충)** in a
honeycomb photo, and a Panda robot arm (ROS 2 Jazzy + MoveIt + RViz) picks the
target off a comb and places it into a grafting tray.

The hard, interesting problem is **not** locating cells (OpenCV does that
deterministically) — it is the **biological classification**: telling apart
`egg`, `young_larva` (the graft target), `old_larva`, and `empty`. That is where
**Gemini** is the decision engine, and where we spent our effort.

---

## Architecture

Two halves, cleanly separated (Vision · Coordinate Transform · Motion · Pick · Place):

```
web/  (Python vision pipeline)                ros2_ws/  (ROS 2 robotics)
  OpenCV cell detection                         beehive_sim  (Gazebo + vision_node)
  → OpenCV reject filter                          live camera frame ─► pipeline
  → visual-priority sort                          selected pixel ─► Stage-4 project
  → BATCHED Gemini classification  ───────────►   onto the comb plane ─► /selected_pose
    (+ RAG reference gallery)                    beehive_transfer  (C++ MoveIt)
  → rank → select one cell                         /selected_pose ─► pick & place
```

The robotics layer imports the **unchanged** `web/` pipeline as a module — there
is one source of truth for the vision logic.

---

## Quick start

### Prerequisites

- Ubuntu 24.04 + **ROS 2 Jazzy** (WSL2 works; the scripts auto-enable software GL).
- The repo checked out somewhere. **Note:** if the path contains a space (e.g.
  `Grafting Project`), the scripts build into a space-free `~/beehive_ws` for you,
  because Gazebo can't load a world path with spaces.

### One-time setup

```bash
./scripts/setup.sh
```

This installs ROS/build deps, installs the vision deps **correctly** (numpy/OpenCV
from apt to match `cv_bridge`'s ABI; only `google-genai`/Pillow/`python-dotenv`
from pip), symlinks the sources into `~/beehive_ws`, builds, and runs a headless
Gazebo + `cv_bridge` sanity check.

### Run the full demo (mock Gemini — no API key)

```bash
./scripts/demo.sh
```

`demo.sh` brings up Gazebo + vision, the MoveIt Panda, and `robot_node`, waits for
a real camera frame, triggers one analysis, and drives the pick & place. Watch the
arm in **RViz**. Gazebo runs **headless by default** on WSL (the GUI starves the
software renderer); pass `BEEHIVE_GAZEBO_GUI=1 ./scripts/demo.sh` to see Gazebo.

For a **live** Gemini run, set a key and disable mock:

```bash
export GEMINI_API_KEY=...                 # a fresh free-tier key
MOCK=false ./scripts/demo.sh
```

### Run the halves separately

```bash
./scripts/run_sim.sh        # Gazebo world + camera + vision_node
./scripts/run_robot.sh      # MoveIt Panda demo + robot_node (waits for move_group)
# then trigger one analysis:
ros2 service call /vision_node/analyze std_srvs/srv/Trigger
```

---

## The scene (robotics sim)

- The comb is a **vertical wall** (0.60 × 0.40 m) standing at `x = 0.5 m` in front
  of the robot, comb face toward the robot.
- A fixed camera sits at the **middle of the Panda** (base center, `z = 0.4 m`)
  looking straight forward; its FOV is sized so the wall exactly fills the
  1024 × 683 frame, so every pixel maps to a point on the comb.
- `vision_node` reads `camera_info` + the TF tree and projects the selected pixel
  onto the wall plane, publishing `/selected_pose` (PoseStamped, `panda_link0`)
  plus an RViz marker at that pose **before** any motion, so the projection is
  visually verifiable.
- `robot_node` consumes only `/selected_pose` and performs a **horizontal,
  wall-facing grasp** → places into a 5 × 5 tray. The pick is *virtual* (a
  collision box attached/detached in the MoveIt planning scene), so no real
  gripper is needed.

Gravity is disabled in the world — it is a camera + visualization sim, and the
unactuated Panda would otherwise collapse on spawn.

---

## How the vision pipeline works

```
photo
  ▼ Stage 1 — OpenCV geometric cell detection      (one cell = one crop)
  ▼ Stage 2 — OpenCV lightweight reject filter      (reflection / flat / sealed; NO biology)
  ▼ sort survivors by OpenCV visual priority
  ▼ Stage 3 — BATCHED Gemini classification         (egg | young_larva | old_larva | empty
  │            + RAG reference gallery prepended)     + confidence + graft_score + reason
  ▼ merge batches → rank (graft_score, confidence) → select one cell
  ▼ Stage 4 — project the selected pixel → base-frame pose → robot pick & place
```

**Division of labor:** OpenCV = *where the cells are* (cheap, deterministic).
Gemini = *what is inside each cell* (all biological reasoning).

Every run writes a `debug/run_<ts>/` folder: annotated detection/filter images, one
clean crop per detected cell under `crops/`, a results CSV, and a summary table.

### What we focused on (the Gemini hard part)

1. **Batched Gemini calls (reliability).** 100+ crops in one multimodal request
   returned empty/blocked responses. We split candidates into batches of ~32–48,
   one call each, and merge by `cell_id`. Batches are independent — a failed batch
   is reported, the rest survive.
2. **OpenCV visual-priority ordering.** Surviving crops are sorted by a
   texture/contrast score so the most promising cells lead the batches.
3. **RAG reference gallery (accuracy).** Curated close-ups of an ideal young larva
   (`web/sample_images/rag/`) are prepended to **every** Gemini batch as in-context
   few-shot ground truth, sharpening the egg vs young vs old distinction. Missing
   folder → clean fallback to the no-reference prompt.

---

## Web dashboard (vision only)

A Flask server + browser dashboard for the vision pipeline on its own:

```bash
cd web
pip install -r requirements.txt    # or use the apt/pip split that setup.sh applies
python app.py                       # http://localhost:5000, upload a honeycomb photo
```

Via Docker:

```bash
cd web && docker build -t beehive-web .
docker run --env-file ../.env -p 5000:5000 beehive-web
```

> On **Docker Desktop (Windows/Mac)** use `-p 5000:5000`, **not** `--network host`
> — host networking doesn't publish ports to the host OS there, so the page would
> refuse to connect.

---

## Configuration (`.env`)

| Variable | What it does |
|---|---|
| `GEMINI_MOCK` | `1` = offline canned results (no API). `0` = real Gemini. |
| `GEMINI_API_KEY` | Required when `GEMINI_MOCK=0`. |
| `GEMINI_MODEL` | Multimodal model (default `gemini-2.5-flash`). |
| `DETECTOR` | `geometry`/`color` (OpenCV stages) or `gemini` (one-pass). |
| `CONFIDENCE_THRESHOLD` | Min confidence to keep a classified larva (default 0.7). |
| `GEMINI_BATCH_SIZE` | Crops per Gemini call (default 48). |
| `GEMINI_REFERENCE_DIR` | Folder of curated young-larva reference crops. |
| `FLASK_PORT` | Web dashboard port (default 5000). |

> Mock verdicts are canned by index, so mock output is for plumbing only — judge
> the classification quality from a **live** run (`GEMINI_MOCK=0` + valid key).

---

## Repository layout

```
beehive_project/
├── scripts/            setup.sh · run_sim.sh · run_robot.sh · demo.sh · lib.sh
├── web/                Flask + OpenCV/Gemini vision pipeline (Docker)
│   ├── pipeline.py  cell_detector.py  gemini_analyzer.py  opencv_*.py  app.py
│   └── sample_images/rag/   curated young-larva reference gallery
└── ros2_ws/src/
    ├── beehive_sim/        Gazebo world (comb wall + camera) + vision_node + coord_transform
    └── beehive_transfer/   C++ robot_node (MoveIt pick & place)
```

---

## Scope & status

- **Vision:** validated live and against real OpenCV on sample combs.
- **Robotics:** a deliberately **constrained simulation** for the one-day timebox —
  RViz/MoveIt with `mock_components` (no `ros2_control` actuation) and a *virtual*
  pick (collision attach/detach, not a real grasp). It proves the end-to-end path
  (live camera → Gemini selection → base-frame pose → collision-aware motion plan).
- **Next:** `ros2_control` actuation in Gazebo, a real gripper (MoveIt `hand`
  group), a physical comb mesh, then a real arm and field trials with beekeepers.

---

## Troubleshooting (WSL)

- **`colcon: command not found`** → `source /opt/ros/jazzy/setup.bash` (setup.sh
  handles this).
- **`No module named catkin_pkg` during build** → a virtualenv is active; the
  scripts drop it for ROS automatically, or run `deactivate` and rebuild.
- **`cv_bridge` SIGSEGV / NumPy 1.x vs 2.x** → a pip/user-site numpy is shadowing
  apt's; re-run `./scripts/setup.sh` (it scrubs `~/.local` numpy/opencv).
- **Gazebo "Unable to find or download file"** → a space in the path; build/run
  from `~/beehive_ws` (setup.sh does this).
- **Camera "no frame" / stalls** → run headless (the demo default) so the GUI
  doesn't starve the software renderer.
- **`robot_node` didn't start** → rebuild `beehive_transfer`; `ros2 service`
  hangs at "waiting for service" mean the node crashed (check `/tmp/beehive/sim.log`).
