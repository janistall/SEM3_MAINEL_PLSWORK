#!/usr/bin/env bash
# Host-side launcher for the read-only Streamable HTTP MCP diagnostic bridge.
# Run this from a normal host terminal (outside Snap confinement).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SWARM_WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
PX4_MSGS_SETUP="${PX4_MSGS_SETUP:-}"
VENV_DIR="${SIM_MCP_VENV:-${SCRIPT_DIR}/.venv}"

if [[ -n "${SNAP:-}" ]]; then
  echo "ERROR: Detected Snap confinement (SNAP=${SNAP}). Run this script in a host terminal outside Snap." >&2
  exit 1
fi

if [[ -f "${ROS_SETUP}" ]]; then
  # Temporarily disable nounset for ROS setup scripts.
  set +u
  # shellcheck disable=SC1090
  source "${ROS_SETUP}"
  set -u
else
  echo "WARNING: ROS 2 setup script not found at ${ROS_SETUP}" >&2
fi

if [[ -n "${PX4_MSGS_SETUP}" && -f "${PX4_MSGS_SETUP}" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${PX4_MSGS_SETUP}"
  set -u
fi

if [[ -f "${WORKSPACE_DIR}/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "${WORKSPACE_DIR}/install/setup.bash"
  set -u
fi

PYTHON_BIN="${SIM_MCP_PYTHON:-python3}"
if [[ -x "${VENV_DIR}/bin/python3" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python3"
fi

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export SWARM_WORKSPACE_DIR="${WORKSPACE_DIR}"

# Default to isolated loopback-only ROS 2 and Gazebo transport for software-only SITL.
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export GZ_IP="${GZ_IP:-127.0.0.1}"
export GZ_PARTITION="${GZ_PARTITION:-swarm}"

exec "${PYTHON_BIN}" -m sim_diagnostics_mcp "$@"
