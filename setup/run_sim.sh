#!/usr/bin/env bash
# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
# Agentic Robotics with ROS 2 — launch the greenhouse sim with all env fixes.
#
# Handles the environment hazards students hit on Ubuntu 24.04:
#   1. VS Code snap terminals inject GTK/locale paths → `gz sim` symbol errors.
#   2. Anaconda python shadows system python and breaks ROS introspection.
#   3. Hybrid Intel+NVIDIA laptops route GL to Intel → Ogre2 hangs (PRIME offload).
#   4. Wayland session → Gazebo Qt GUI hangs on a blank window (force xcb/XWayland).
#
# Usage:
#   ./setup/run_sim.sh                                  # default greenhouse world
#   ./setup/run_sim.sh world:=greenhouse_2026_lite.sdf  # CPU-only laptop
#   ./setup/run_sim.sh y:=-1.0                          # spawn in a different aisle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../raise_ros2_ws" && pwd)"

SNAP_VARS=(
  GTK_PATH GTK_EXE_PREFIX GTK_IM_MODULE_FILE GIO_MODULE_DIR LOCPATH
  GSETTINGS_SCHEMA_DIR GSETTINGS_SCHEMA_DIR_CONDA_BACKUP
  XDG_DATA_HOME XDG_CONFIG_HOME XDG_CACHE_HOME
  VSCODE_GIT_ASKPASS_NODE VSCODE_GIT_ASKPASS_MAIN VSCODE_GIT_IPC_HANDLE
  GIT_ASKPASS VSCODE_INJECTION VSCODE_IPC_HOOK_CLI ELECTRON_RUN_AS_NODE
)
CONDA_VARS=(CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE)
unset_args=(); for v in "${SNAP_VARS[@]}" "${CONDA_VARS[@]}"; do unset_args+=(-u "$v"); done

PRIME_VARS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  PRIME_VARS=(__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only)
fi

QT_VARS=()
if [[ "${XDG_SESSION_TYPE:-}" == "wayland" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  QT_VARS=(QT_QPA_PLATFORM=xcb)
fi

strip_paths() { echo "${1:-}" | tr ':' '\n' | grep -vE '/snap/|anaconda' | paste -sd:; }
strip_snap()  { echo "${1:-}" | tr ':' '\n' | grep -v '/snap/'           | paste -sd:; }

if [[ ! -f "${WS_DIR}/install/setup.bash" ]]; then
  cat >&2 <<EOF
✗ Workspace not built at ${WS_DIR}.
  Run the installer first:  ${SCRIPT_DIR}/install.sh
EOF
  exit 1
fi

exec env "${unset_args[@]}" \
    PATH="$(strip_paths "${PATH:-/usr/bin:/bin}")" \
    LD_LIBRARY_PATH="$(strip_snap "${LD_LIBRARY_PATH:-}")" \
    XDG_DATA_DIRS="$(strip_snap "${XDG_DATA_DIRS:-/usr/local/share:/usr/share}")" \
    DISPLAY="${DISPLAY:-:0}" \
    XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}" \
    HOME="${HOME}" USER="${USER}" \
    "${PRIME_VARS[@]}" "${QT_VARS[@]}" WS_DIR="${WS_DIR}" \
  bash -c '
    source /opt/ros/jazzy/setup.bash
    source "${WS_DIR}/install/setup.bash"
    exec ros2 launch raise2026_bringup sim.launch.py "$@"
  ' bash "$@"
