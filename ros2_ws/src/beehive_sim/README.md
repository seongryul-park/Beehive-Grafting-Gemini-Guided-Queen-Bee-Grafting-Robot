# beehive_sim — Gazebo bringup + live camera → vision pipeline

Stage 1–3 of the robotics integration. It puts the **existing** beehive vision
pipeline (OpenCV → Gemini → ranking) behind a Gazebo camera so the robotics
system consumes a *live* rendered image instead of a pre-recorded file. The
vision code in `web/` is **not modified** — it is imported as a module.

```
Gazebo camera ─/beehive/camera/image─► vision_node ─► (existing pipeline)
                                            │                  │
                                            │            selected pixel
                                            │                  │
                                            │   Stage 4: camera_info + TF tree
                                            │     pixel → ray → comb plane →
                                            │        point in base frame
                                            │                  │
              ┌─────────────┬─────────┼────────────┬──────────────┐
         ~/annotated    ~/result  /selected_pose /selected_pose_  /selected_pixel
       (live overlay)   (JSON)    (PoseStamped,   marker (RViz     (raw pixel,
                                   BASE frame)     sphere)          debug)
```

The robot stack (RViz / MoveIt / pick-and-place) consumes **`/selected_pose`**
directly. `/selected_pose_marker` is published at the same pose *before any
motion* so the projection accuracy can be checked visually in RViz.

## What's here

| Path | Purpose |
|---|---|
| `worlds/beehive.sdf` | World: sun, ground, table, honeycomb plane, fixed camera |
| `models/honeycomb_plane/` | Static plane textured with `web/sample_images/comb.jpg` |
| `models/work_surface/` | Static table (top at z = 0) |
| `beehive_sim/vision_core.py` | ROS-free wrapper: import pipeline, run on a frame, annotate |
| `beehive_sim/coord_transform.py` | ROS-free Stage-4 math: pixel → plane → base-frame point |
| `beehive_sim/vision_node.py` | ROS node: camera → pipeline → base-frame pose + marker |
| `launch/sim_bringup.launch.py` | Gazebo + camera bridge + vision node + static TF (+ optional robot/rqt) |

## Requirements

ROS 2 **Jazzy**, Gazebo **Harmonic**, and:

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-cv-bridge ros-jazzy-rqt-image-view
# the vision pipeline's Python deps must be importable by ROS:
pip install -r ../../../web/requirements.txt   # opencv-python-headless, pillow, google-genai, numpy
```

## Build

```bash
cd ros2_ws
colcon build --packages-select beehive_sim
source install/setup.bash
```

## Run (the thing to verify first)

```bash
# point the node at the unchanged vision code
export BEEHIVE_WEB_DIR=$(pwd)/../web        # abs path to beehive_project/web

ros2 launch beehive_sim sim_bringup.launch.py
# (mock Gemini by default — no API key needed; real OpenCV detection still runs)
```

Then, in a second terminal:

```bash
source install/setup.bash

# watch the live camera + overlay
ros2 run rqt_image_view rqt_image_view /vision_node/annotated

# run the pipeline on the current live frame
ros2 service call /vision_node/analyze std_srvs/srv/Trigger

# inspect outputs
ros2 topic echo /vision_node/result --once
ros2 topic echo /vision_node/selected_pixel --once
```

## Verify Stage 1 is working

1. Gazebo shows the table + honeycomb plane, and the camera is looking down at it.
2. `ros2 topic hz /beehive/camera/image` shows ~15 Hz — the camera is live.
3. `rqt_image_view` on `/vision_node/annotated` shows the live comb.
4. After the `analyze` service call, the overlay shows detected cells, amber
   candidates, and one green **selected** cell with a crosshair; the service
   returns e.g. `cells=145 candidates=87 selected=81 pixel=(60, 516)`.

Offline sanity check of the exact import path (no ROS/Gazebo needed):

```bash
python3 - <<'PY'
import sys; sys.path.insert(0, "beehive_sim")
import cv2, vision_core as vc
frame = cv2.imread("../../../web/sample_images/comb.jpg")   # stand-in for a camera frame
vp = vc.VisionPipeline("../../../web", mock=True, detector="geometry")
res = vp.analyze_bgr(frame)
print("cells", len(res["cells"]), "candidates", len(res["candidates"]),
      "selected", res["selected"], "pixel", vc.selected_pixel(res))
PY
```

## Useful launch args

| Arg | Default | Meaning |
|---|---|---|
| `web_dir` | `$BEEHIVE_WEB_DIR` | Path to `beehive_project/web` (required for analysis) |
| `mock` | `true` | Mock Gemini. Set `false` (+ `GEMINI_API_KEY`) for real classification |
| `detector` | `geometry` | OpenCV detector: `geometry` / `opencv` / `color` |
| `spawn_robot` | `true` | Spawn Panda for visualization (skipped if description missing) |
| `rqt` | `false` | Auto-open `rqt_image_view` on the annotated topic |

## Configuration notes

- **Texture / which comb:** replace
  `models/honeycomb_plane/materials/textures/comb.jpg` with another image from
  `web/sample_images/` (keep the filename, or edit `model.sdf`), then rebuild.
- **Camera framing:** the camera is at `(0.5, 0, 0.55)` looking straight down
  with a 60° FOV in `worlds/beehive.sdf`. Adjust the pose there if the comb
  doesn't fill the frame.
- **Continuous mode:** set the node param `auto_analyze:=true` (and
  `auto_period`) to analyze periodically instead of on the service. Only sensible
  in `mock` mode — live Gemini would be called every period.

## Stage 4 — pixel → base-frame pose (done)

After the pipeline picks a cell, `vision_node` projects its pixel to a 3D pose:

1. `camera_info` gives the pinhole intrinsics `K` (read live off the bridge).
2. `coord_transform.backproject_pixel` makes a ray in the camera optical frame.
3. The TF tree (`base_frame ← honeycomb_camera_optical_frame`, from the static
   transforms in the launch) rotates the ray into the robot base frame.
4. The ray is intersected with the honeycomb plane (`plane_z`) → a 3D point.
5. `/selected_pose` (PoseStamped, base frame) is published with a top-down grasp
   orientation, plus `/selected_pose_marker` for RViz.

No image-specific coordinates are hardcoded — only the fixed scene geometry
(camera pose, plane height) lives in the world/launch, exactly as assumed.

**Verify the projection in RViz (before any motion):**

```bash
rviz2
# Fixed Frame: panda_link0   (or 'world')
# add: TF, MarkerArray on /selected_pose_marker, Camera/Image on /vision_node/annotated
ros2 service call /vision_node/analyze std_srvs/srv/Trigger
# the green sphere marker should sit on the comb at the selected cell.
ros2 topic echo /selected_pose --once
```

Sanity numbers (camera at z=0.50, FOV 0.586, plane at z=0.003): the image center
maps to `(0.500, 0.000, 0.003)` and the comb corners to `x∈[0.35,0.65]`,
`y∈[-0.10,0.10]` in `panda_link0` — i.e. the 0.30×0.20 m plane fills the frame.

## Not in this stage (by design)

Full Gazebo `ros2_control` actuation of the arm is out of scope here; motion
planning (Stages 5–7) runs through MoveIt and consumes `/selected_pose` directly
— see `beehive_transfer/robot_node` and `beehive.launch.py`.
