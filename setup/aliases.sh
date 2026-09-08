# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
# Agentic Robotics with ROS 2 — student convenience aliases.
# install.sh sources this from ~/.bashrc, so every new terminal is ready:
#   • ROS 2 + this workspace are sourced automatically
#   • API keys from raise_ros2_ws/src/.env are loaded (for VLM/agent labs)
#   • one-word helpers: raise-sim, raise-stop, raise-check, raise-teleop
#   • skill-server shortcuts: gripper_server, navigation_server, ...

# paths derived from THIS file's location (portable; symlink-safe)
_ARR_SETUP_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
_ARR_ROOT="$(cd "${_ARR_SETUP_DIR}/.." && pwd)"
export RAISE_WS="${_ARR_ROOT}/raise_ros2_ws"
RAISE_ENV="${RAISE_WS}/src/.env"

# ── source ROS 2 Jazzy + this workspace overlay ─────────────────────────────
[ -f /opt/ros/jazzy/setup.bash ] && source /opt/ros/jazzy/setup.bash
[ -f "${RAISE_WS}/install/setup.bash" ] && source "${RAISE_WS}/install/setup.bash"

# ── load API keys (OPENAI_API_KEY, ...) if the student added a .env ─────────
[ -f "${RAISE_ENV}" ] && { set -a; . "${RAISE_ENV}"; set +a; }

# ── one-word helpers ────────────────────────────────────────────────────────
alias raise-sim="${_ARR_SETUP_DIR}/run_sim.sh"       # start the greenhouse sim
alias raise-check="${_ARR_SETUP_DIR}/check_setup.sh" # is my machine ready?
alias raise-teleop="ros2 run raise2026_teleop teleop_keyboard"  # drive: w/s a/d SPACE

# stop everything the sim started (graceful SIGINT, then SIGKILL leftovers)
raise-stop() {
  local patterns=('ros2 launch raise2026' 'gz sim' 'parameter_bridge'
                  'ros_gz_sim/create' 'robot_state_publisher.*launch_params'
                  'raise2026_teleop.teleop_phone' 'raise2026_teleop.teleop_keyboard')
  local p pid pids=()
  for p in "${patterns[@]}"; do
    while read -r pid; do [ -n "$pid" ] && [ "$pid" != "$$" ] && pids+=("$pid"); done \
      < <(pgrep -f "$p" 2>/dev/null || true)
  done
  pids=($(printf '%s\n' "${pids[@]}" | sort -u | grep -v '^$' || true))
  if [ "${#pids[@]}" -eq 0 ]; then echo "raise-stop: no running sim found."; return 0; fi
  echo "raise-stop: stopping ${pids[*]}"; kill -INT "${pids[@]}" 2>/dev/null || true; sleep 2
  kill -9 "${pids[@]}" 2>/dev/null || true; echo "raise-stop: done."
}

# ── skill-server shortcuts (run a tool server in the current terminal) ──────
alias gripper_server="ros2 run raise2026_tools gripper_server"
alias navigation_server="ros2 run raise2026_tools navigation_server"
alias move_to_pose_server="ros2 run raise2026_tools move_to_pose_server"
alias detector_server="ros2 run raise2026_tools detector_server"
alias inspector_server="ros2 run raise2026_tools inspector_server"
alias grasp_server="ros2 run raise2026_tools grasp_server"
