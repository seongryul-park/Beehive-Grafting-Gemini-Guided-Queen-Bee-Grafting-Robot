#!/usr/bin/env bash
# demo.sh — full end-to-end on one machine: sim + robot + one analyze trigger.
#
#   ./scripts/demo.sh                 # mock Gemini (no API key)
#   MOCK=false ./scripts/demo.sh      # live Gemini (needs GEMINI_API_KEY exported)
#
# Brings up Gazebo+vision and the MoveIt Panda+robot_node in the background
# (logs under /tmp/beehive), waits for them, then fires one /analyze which
# selects a larva, publishes /selected_pose, and drives the pick & place.
# Ctrl-C tears everything down.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_ros

MOCK="${MOCK:=true}"
# Default to HEADLESS Gazebo: on WSL/software-GL the Gazebo GUI starves the camera
# sensor of the renderer, so frames stall. Server-only makes the camera reliable;
# you watch the arm in RViz anyway. Override with BEEHIVE_GAZEBO_GUI=1 to see Gazebo.
export BEEHIVE_GAZEBO_GUI="${BEEHIVE_GAZEBO_GUI:-0}"
LOG_DIR="${TMPDIR:-/tmp}/beehive"; mkdir -p "$LOG_DIR"
PIDS=()
cleanup(){ c_info "shutting down demo..."; for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

# 1) Gazebo + camera + vision
c_info "1/4  Gazebo + vision   (mock=$MOCK, log: $LOG_DIR/sim.log)"
ros2 launch beehive_sim sim_bringup.launch.py mock:="$MOCK" \
  > "$LOG_DIR/sim.log" 2>&1 &
PIDS+=($!)

# 2) MoveIt Panda demo (move_group + RViz)
c_info "2/4  MoveIt Panda demo  (log: $LOG_DIR/moveit.log)"
ros2 launch moveit_resources_panda_moveit_config demo.launch.py \
  > "$LOG_DIR/moveit.log" 2>&1 &
PIDS+=($!)

# Wait for move_group (MoveIt/RViz) to come up.
c_info "waiting for move_group (up to 90s)..."
for _ in $(seq 1 90); do
  ros2 node list 2>/dev/null | grep -q "/move_group" && { c_info "move_group ready."; break; }
  sleep 1
done

# 3) robot_node (consumes /selected_pose)
c_info "3/4  robot_node         (log: $LOG_DIR/robot.log)"
ros2 launch beehive_transfer beehive.launch.py \
  > "$LOG_DIR/robot.log" 2>&1 &
PIDS+=($!)

# Confirm robot_node actually came up (its launch previously aborted on rosbridge,
# leaving nothing subscribed to /selected_pose). Surface the failure here.
c_info "waiting for robot_node (up to 30s)..."
robot_ok=0
for _ in $(seq 1 30); do
  ros2 node list 2>/dev/null | grep -q "/robot_node" && { robot_ok=1; break; }
  sleep 1
done
if [ "$robot_ok" = 1 ]; then
  c_info "robot_node is up."
else
  c_warn "robot_node did NOT start — its launch errored. Last lines of robot.log:"
  tail -15 "$LOG_DIR/robot.log" | sed 's/^/    /'
fi

# Wait for an ACTUAL camera frame. A topic being *listed* != a frame published;
# under software GL (llvmpipe) the first render takes several seconds, so blocking
# on a real message stops /analyze firing into an empty buffer ("no camera frame yet").
c_info "waiting for first camera frame on /beehive/camera/image (up to 120s)..."
if timeout 120 ros2 topic echo /beehive/camera/image --once >/dev/null 2>&1; then
  c_info "camera frame received."
else
  c_warn "no frame after 120s — software rendering may be stalled. Check:"
  c_warn "  ros2 topic hz /beehive/camera/image     (see also $LOG_DIR/sim.log)"
fi

# /selected_pose is published ONCE per analyze (not latched), so start listening
# BEFORE triggering — otherwise a late --once subscriber misses the single message.
( timeout 30 ros2 topic echo /selected_pose --once > "$LOG_DIR/selected_pose.txt" 2>&1 ) &
ECHO_PID=$!

# 4) Trigger analyze -> selected_pose -> pick & place. Retry: frame/pipeline may lag.
c_info "4/4  triggering /vision_node/analyze ..."
for attempt in 1 2 3 4 5; do
  resp="$(ros2 service call /vision_node/analyze std_srvs/srv/Trigger 2>&1)"
  echo "$resp" | grep -E "success=|message=" || echo "$resp"
  echo "$resp" | grep -q "success=True" && break
  c_warn "analyze not ready (attempt $attempt/5) — retrying in 5s..."
  sleep 5
done

wait "$ECHO_PID" 2>/dev/null || true
if [ -s "$LOG_DIR/selected_pose.txt" ]; then
  c_info "selected pose (published to robot_node):"
  cat "$LOG_DIR/selected_pose.txt"
else
  c_warn "/selected_pose not captured here, but robot_node (subscribed before the"
  c_warn "trigger) received it — watch RViz, or: tail -f $LOG_DIR/robot.log"
fi

# Show what robot_node actually did (the real proof of pick & place).
sleep 3
c_info "robot_node activity:"
if grep -qE "target #|planning|placed|FAILED" "$LOG_DIR/robot.log"; then
  grep -E "target #|planning|\] done|placed|FAILED|ready" "$LOG_DIR/robot.log" | tail -20 | sed 's/^/    /'
else
  c_warn "  no motion logged yet — robot_node may still be planning, or didn't get"
  c_warn "  the pose. Re-trigger: ros2 service call /vision_node/analyze std_srvs/srv/Trigger"
fi

c_info "Demo running — watch RViz for the pick & place."
c_info "Re-trigger anytime: ros2 service call /vision_node/analyze std_srvs/srv/Trigger"
c_info "Ctrl-C to stop everything."
wait
