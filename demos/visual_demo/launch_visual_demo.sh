#!/usr/bin/env bash
# ============================================================================
# Standalone 10-Drone Gazebo Visual Research Demonstration Launcher
#
# Scripted illustration of proposed LiDAR and AI-assisted navigation.
# Completely disconnected from PX4, ROS 2 flight-control topics, MAVLink,
# MicroXRCEAgent, and DDS. Reuses existing arena appearance, obstacles,
# destination markers, and x500_base drone meshes with manually authored
# prerecorded keyframes, scripted perception graphics, and a live dashboard.
# ============================================================================
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX4_DIR="${PX4_DIR:-/home/siddharthg/PX4-Autopilot}"
WORLD_FILE="$DEMO_DIR/worlds/swarm_visual_demo.sdf"
TRAJ_FILE="$DEMO_DIR/trajectories.json"
MODELS_DIR="$PX4_DIR/Tools/simulation/gz/models"

if [[ ! -f "$WORLD_FILE" || ! -f "$TRAJ_FILE" ]]; then
  echo "Generating visual demo world and prerecorded keyframes..."
  python3 "$DEMO_DIR/generate_demo.py"
fi

if [[ ! -d "$MODELS_DIR/x500_base" ]]; then
  echo "ERROR: x500_base meshes not found at $MODELS_DIR/x500_base" >&2
  echo "Set PX4_DIR=/path/to/PX4-Autopilot and re-run." >&2
  exit 1
fi

if ! command -v gz >/dev/null 2>&1; then
  echo "ERROR: Gazebo Harmonic ('gz') not found on PATH." >&2
  exit 1
fi

# Strictly Gazebo-only loopback & dedicated partition (zero ROS / DDS / PX4 / MAVLink).
unset ROS_DISTRO ROS_VERSION ROS_DOMAIN_ID ROS_LOCALHOST_ONLY FASTDDS_DEFAULT_PROFILES_FILE
export GZ_IP="127.0.0.1"
export GZ_PARTITION="swarm_visual_demo"
export GZ_SIM_RESOURCE_PATH="$MODELS_DIR:$DEMO_DIR/worlds"

GZ_PID=""
cleanup() {
  if [[ -n "$GZ_PID" ]] && kill -0 "$GZ_PID" 2>/dev/null; then
    kill -INT "$GZ_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$GZ_PID" 2>/dev/null || break
      sleep 0.15
    done
    kill -TERM "$GZ_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

gz sim -r "$WORLD_FILE" --verbose=0 &
GZ_PID=$!

python3 "$DEMO_DIR/play_demo.py" --trajectories "$TRAJ_FILE" --gz-pid "$GZ_PID" "$@"
