#!/usr/bin/env bash
# run_sim.sh — Gazebo world + camera bridge + vision_node (Stage 1-4).
# Pass-through launch args, e.g.:
#   ./scripts/run_sim.sh                 # mock Gemini (no key)
#   ./scripts/run_sim.sh mock:=false     # live Gemini (needs GEMINI_API_KEY)
#   ./scripts/run_sim.sh rqt:=true       # auto-open the annotated camera view
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
activate_ros

c_info "Launching Gazebo + vision (web_dir=$BEEHIVE_WEB_DIR)"
c_info "In another terminal, trigger a run:"
c_info "  ros2 service call /vision_node/analyze std_srvs/srv/Trigger"
exec ros2 launch beehive_sim sim_bringup.launch.py "$@"
