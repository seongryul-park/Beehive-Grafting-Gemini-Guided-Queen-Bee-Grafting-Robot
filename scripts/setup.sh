#!/usr/bin/env bash
# setup.sh — one-time bootstrap: deps + space-free workspace + build.
# Run from anywhere:  ./scripts/setup.sh
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

# ament/colcon must build with SYSTEM python (needs catkin_pkg). Drop any active
# venv for this process so `python3` doesn't resolve into .venv (that causes the
# 'No module named catkin_pkg' build failure).
deactivate_venv

c_info "Project   : $PROJECT_DIR"
c_info "ROS distro: $ROS_DISTRO"
c_info "Workspace : $WS_DIR  (space-free build location)"

# 1) ROS present?
if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
  c_err "ROS 2 '$ROS_DISTRO' is not installed (/opt/ros/$ROS_DISTRO missing)."
  c_err "Install ros-$ROS_DISTRO-desktop first, then re-run this script."
  exit 1
fi
source "/opt/ros/$ROS_DISTRO/setup.bash"

# 2) colcon present? (the right package — NOT 'apt install colcon')
if ! command -v colcon >/dev/null 2>&1; then
  c_info "Installing colcon build tools..."
  sudo apt update
  sudo apt install -y python3-colcon-common-extensions
fi

# ament build deps in SYSTEM python (catkin_pkg/empy) — the cmake packages need them.
if ! python3 -c "import catkin_pkg" >/dev/null 2>&1; then
  c_info "Installing ROS build deps (catkin_pkg/empy)..."
  sudo apt install -y python3-catkin-pkg python3-empy || \
    python3 -m pip install catkin_pkg empy lark --break-system-packages
fi

# 3) Vision pipeline Python deps into the SYSTEM python (the one ROS nodes use).
#
#    CRITICAL ABI RULE: numpy and OpenCV must come from Ubuntu APT, never pip.
#    ROS Jazzy's cv_bridge is compiled against the system numpy/opencv. pip's
#    numpy 2.x + opencv-python-headless shadow them (they land in
#    /usr/local/lib .../dist-packages, which precedes /usr/lib) and crash
#    cv_bridge inside imgmsg_to_cv2 -> vision_node dies before advertising
#    /analyze ("waiting for service to become available"). So:
#      - numpy / opencv  -> apt  (python3-numpy, python3-opencv)
#      - everything else -> pip  (only what Ubuntu doesn't ship for ROS)
python3 -m pip --version >/dev/null 2>&1 || sudo apt install -y python3-pip

c_info "Installing numpy/opencv from APT (ABI-matched to cv_bridge)..."
sudo apt install -y python3-numpy python3-opencv

# Remove EVERY pip-installed numpy/opencv copy that shadows the apt ones — this
# includes USER-SITE (~/.local/lib/.../site-packages), which has HIGHER import
# priority than both apt (/usr/lib) and /usr/local. A leftover user-site numpy
# 2.x is exactly what SIGSEGVs cv_bridge at imgmsg_to_cv2. Loop because there can
# be several copies on sys.path (system + user).
c_info "Removing pip/user numpy/opencv shadows (if present)..."
for pkg in numpy opencv-python opencv-python-headless; do
  for _ in 1 2 3; do
    python3 -m pip uninstall -y "$pkg" >/dev/null 2>&1 || break
  done
done
# Belt-and-suspenders: physically drop any user-site copies pip didn't catch.
rm -rf "$HOME"/.local/lib/python3*/site-packages/numpy* \
       "$HOME"/.local/lib/python3*/site-packages/cv2* \
       "$HOME"/.local/lib/python3*/site-packages/opencv* 2>/dev/null || true

# pip-only deps Ubuntu doesn't provide for ROS. --upgrade-strategy only-if-needed
# stops a transitive dep from silently pulling numpy 2.x back in.
c_info "Installing pip-only deps (google-genai, Pillow, python-dotenv, flask)..."
python3 -m pip install --break-system-packages --upgrade-strategy only-if-needed \
  google-genai pillow python-dotenv flask
# (cv_bridge ABI is verified after the ROS packages are installed — see below.)

# 4) ROS package deps: MoveIt, Gazebo bridge, camera view, rosbridge.
c_info "Installing ROS package deps (sudo apt)..."
sudo apt install -y \
  "ros-$ROS_DISTRO-moveit" \
  "ros-$ROS_DISTRO-moveit-resources-panda-moveit-config" \
  "ros-$ROS_DISTRO-ros-gz" \
  "ros-$ROS_DISTRO-cv-bridge" \
  "ros-$ROS_DISTRO-rqt-image-view" \
  "ros-$ROS_DISTRO-rosbridge-server" \
  "ros-$ROS_DISTRO-rviz-visual-tools" \
  "ros-$ROS_DISTRO-moveit-visual-tools" \
  libcurl4-openssl-dev \
  || c_warn "some apt packages failed — verify names for your distro."

# 4b) Sanity: print NumPy version and PROVE cv_bridge imports (the exact thing
#     that was crashing vision_node). Runs AFTER cv-bridge is apt-installed.
c_info "Verifying vision deps + cv_bridge ABI..."
python3 - <<'PYEOF'
import numpy, cv2
# print WHERE numpy loads from — must be apt (/usr/lib), not ~/.local or /usr/local
print(f"[beehive] NumPy {numpy.__version__} from {numpy.__file__}")
print(f"[beehive] OpenCV {cv2.__version__}")
assert int(numpy.__version__.split('.')[0]) < 2, (
    "NumPy 2.x still active -- a user-site/pip copy is shadowing apt. "
    "Fix: python3 -m pip uninstall -y numpy; "
    "rm -rf ~/.local/lib/python3*/site-packages/numpy*  then re-run setup.sh")
assert ".local" not in numpy.__file__, (
    f"NumPy is loading from user-site ({numpy.__file__}) -- remove it: "
    "rm -rf ~/.local/lib/python3*/site-packages/numpy*")
from cv_bridge import CvBridge            # the import that SIGSEGV'd at imgmsg_to_cv2
CvBridge()
import google.genai, PIL                  # noqa: F401
print("[beehive] cv_bridge import OK -- vision deps OK")
PYEOF

# 5) Space-free workspace: symlink the sources, build there.
mkdir -p "$WS_DIR"
if [ ! -e "$WS_DIR/src" ]; then
  ln -s "$SRC_DIR" "$WS_DIR/src"
  c_info "linked $WS_DIR/src -> $SRC_DIR"
fi
cd "$WS_DIR"
# A prior build (with a venv active) caches the venv's python in CMakeCache.txt,
# so it keeps failing with 'No module named catkin_pkg' even after we drop the
# venv. Wipe any build cache that pins a venv interpreter, and force system python.
SYS_PY="$(command -v python3)"
if grep -rqsiE 'PYTHON_EXECUTABLE.*\.venv|VIRTUAL_ENV' "$WS_DIR/build" 2>/dev/null; then
  c_warn "stale build cache pins a virtualenv python — cleaning build/install/log."
  rm -rf "$WS_DIR/build" "$WS_DIR/install" "$WS_DIR/log"
fi
c_info "Building beehive_sim + beehive_transfer (python=$SYS_PY)..."
colcon build --packages-select beehive_sim beehive_transfer \
  --cmake-args "-DPython3_EXECUTABLE=$SYS_PY" "-DPYTHON_EXECUTABLE=$SYS_PY"

# 6) Rendering env (software GL on WSL) + headless Gazebo smoke-test, so a judge
#    knows the sim actually boots on THIS machine before running the full demo.
source "$WS_DIR/install/setup.bash"
setup_render_env
WORLD="$WS_DIR/install/beehive_sim/share/beehive_sim/worlds/beehive.sdf"
export GZ_SIM_RESOURCE_PATH="$WS_DIR/install/beehive_sim/share/beehive_sim/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
LOG_DIR="${TMPDIR:-/tmp}/beehive"; mkdir -p "$LOG_DIR"
SMOKE="$LOG_DIR/gz_smoke.log"
c_info "Smoke-testing Gazebo headless (server only, ~10s)..."
timeout 10 gz sim -s -r -v 2 "$WORLD" > "$SMOKE" 2>&1 || true
if grep -qiE 'Unable to find or download|Fuel.*failed' "$SMOKE"; then
  c_err  "Gazebo could not resolve the world/models — usually a space in the path."
  c_err  "Build/run from \$WS_DIR ($WS_DIR), not the Desktop. See $SMOKE"
elif grep -qiE 'Failed to create.*(render|ogre)|Unable to create the rendering|EGL' "$SMOKE"; then
  c_warn "Gazebo render engine failed to start — GPU/GL issue."
  c_warn "Re-run with: BEEHIVE_FORCE_SOFTWARE_GL=1 ./scripts/setup.sh   (see $SMOKE)"
else
  c_info "Gazebo smoke-test OK (world + models load, renderer starts)."
fi

c_info "Done. Next:"
c_info "  ./scripts/run_sim.sh      # Gazebo + camera + vision"
c_info "  ./scripts/run_robot.sh    # MoveIt Panda + robot pick&place"
c_info "  ./scripts/demo.sh         # full end-to-end (mock Gemini, no key)"
