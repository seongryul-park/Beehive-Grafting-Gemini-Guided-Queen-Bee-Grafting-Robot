#!/usr/bin/env bash
# lib.sh — shared config + helpers for the beehive grafting scripts.
# Sourced by setup.sh / run_sim.sh / run_robot.sh / demo.sh. Not run directly.

# ROS 2 distribution (override: ROS_DISTRO=kilted ./scripts/setup.sh)
: "${ROS_DISTRO:=jazzy}"

# Resolve paths from THIS file's location so the scripts work wherever the
# project lives — including paths that contain spaces (e.g. "Grafting Project").
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$_LIB_DIR/.." && pwd)"     # .../beehive_project
SRC_DIR="$PROJECT_DIR/ros2_ws/src"            # package sources (may contain spaces)
WEB_DIR="$PROJECT_DIR/web"                    # vision pipeline

# Build/run from a SPACE-FREE path in $HOME. Gazebo's gz_args splits the world
# path on spaces, so it cannot live under "Grafting Project". We symlink the
# sources into $WS_DIR and build there; the install/ tree then has no spaces.
: "${WS_DIR:=$HOME/beehive_ws}"

c_info(){ printf '\033[1;34m[beehive]\033[0m %s\n' "$*"; }
c_warn(){ printf '\033[1;33m[beehive]\033[0m %s\n' "$*"; }
c_err(){  printf '\033[1;31m[beehive]\033[0m %s\n' "$*" >&2; }

# Gazebo's ogre2 camera needs a GL context. On WSL (and headless/VM machines)
# hardware GL usually can't initialize, so the sim dies or renders nothing.
# Force the llvmpipe software renderer there: slow but works on ANY machine,
# which is what we want so a judge can run the demo without GPU setup.
# Override by exporting BEEHIVE_FORCE_SOFTWARE_GL=0 (or =1 to force on non-WSL).
setup_render_env(){
  local force="${BEEHIVE_FORCE_SOFTWARE_GL:-auto}"
  local is_wsl=0
  grep -qiE 'microsoft|wsl' /proc/version 2>/dev/null && is_wsl=1
  if [ "$force" = "1" ] || { [ "$force" = "auto" ] && [ "$is_wsl" = "1" ]; }; then
    export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
    export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
    local why="forced"; [ "$is_wsl" = "1" ] && why="WSL detected"
    c_info "software GL enabled ($why): LIBGL_ALWAYS_SOFTWARE=1, GALLIUM_DRIVER=llvmpipe"
  fi
}

# ROS/ament must build AND run with the SYSTEM python: it needs catkin_pkg/empy,
# and our vision deps were installed there (--break-system-packages). An active
# virtualenv hijacks `python3` and ament fails with 'No module named catkin_pkg'.
# Drop any active venv from THIS process only (the user's shell is untouched).
deactivate_venv(){
  [ -z "${VIRTUAL_ENV:-}" ] && return 0
  c_warn "active virtualenv ignored for ROS: $VIRTUAL_ENV (using system python)"
  local clean="" p IFS=':'
  for p in $PATH; do
    [ "$p" = "$VIRTUAL_ENV/bin" ] && continue
    clean="${clean:+$clean:}$p"
  done
  export PATH="$clean"
  unset VIRTUAL_ENV PYTHONHOME
}

# Source ROS + the built workspace, and export the web dir for vision_node.
activate_ros(){
  deactivate_venv
  if [ ! -f "/opt/ros/$ROS_DISTRO/setup.bash" ]; then
    c_err "ROS 2 '$ROS_DISTRO' not found at /opt/ros/$ROS_DISTRO. Run scripts/setup.sh first."
    return 1
  fi
  # shellcheck disable=SC1090
  source "/opt/ros/$ROS_DISTRO/setup.bash"
  if [ -f "$WS_DIR/install/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "$WS_DIR/install/setup.bash"
  else
    c_warn "workspace not built ($WS_DIR/install missing). Run scripts/setup.sh first."
  fi
  # Used as a Python import path by vision_node — spaces are fine here.
  export BEEHIVE_WEB_DIR="$WEB_DIR"
  setup_render_env
}
