#!/usr/bin/env bash
# run_robot.sh — MoveIt Panda demo (RViz) + robot_node pick&place.
#
# robot_node needs move_group running first (else it dies after ~10s), so this
# script starts the MoveIt demo, WAITS for move_group, then launches our node.
# Pass-through args go to beehive.launch.py.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_ros

LOG_DIR="${TMPDIR:-/tmp}/beehive"; mkdir -p "$LOG_DIR"

c_info "Starting MoveIt Panda demo + RViz  (log: $LOG_DIR/moveit_demo.log)"
ros2 launch moveit_resources_panda_moveit_config demo.launch.py \
  > "$LOG_DIR/moveit_demo.log" 2>&1 &
MOVEIT_PID=$!
cleanup(){ c_info "stopping MoveIt demo..."; kill "$MOVEIT_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

c_info "Waiting for /move_group (up to 60s)..."
for _ in $(seq 1 60); do
  if ros2 node list 2>/dev/null | grep -q "/move_group"; then
    c_info "move_group is up."; break
  fi
  if ! kill -0 "$MOVEIT_PID" 2>/dev/null; then
    c_err "MoveIt demo exited early — see $LOG_DIR/moveit_demo.log"; exit 1
  fi
  sleep 1
done

c_info "Starting robot_node (subscribes /selected_pose)..."
ros2 launch beehive_transfer beehive.launch.py "$@"
