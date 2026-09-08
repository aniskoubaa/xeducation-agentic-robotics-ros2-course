#!/usr/bin/env bash
# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
# Agentic Robotics with ROS 2 — machine readiness check (Ubuntu 24.04 + Jazzy).
# Verifies your machine can run the emulated RAISE-Bot and prints a pass/fail report.
# Usage:  ./setup/check_setup.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/../raise_ros2_ws" 2>/dev/null && pwd || echo "${SCRIPT_DIR}/../raise_ros2_ws")"

PASS=0; WARN=0; FAIL=0
ok()   { echo "  ✓ $*"; PASS=$((PASS+1)); }
warn() { echo "  ⚠ $*"; WARN=$((WARN+1)); }
bad()  { echo "  ✗ $*"; FAIL=$((FAIL+1)); }

echo; echo "════════════════════════════════════════════════════════"
echo "  Agentic Robotics with ROS 2 — machine readiness check"
echo "════════════════════════════════════════════════════════"

# ── 1. Operating system (Ubuntu 24.04 ONLY) ─────────────────────────────────
echo; echo "[1/6] Operating system"
if [[ "$(uname -s)" != "Linux" ]]; then
  bad "Not Linux — this course requires native Ubuntu 24.04."; echo; echo "SUMMARY: ✗ NOT READY"; exit 1
fi
os_id=unknown; os_version=unknown
[[ -r /etc/os-release ]] && { source /etc/os-release; os_id="${ID:-unknown}"; os_version="${VERSION_ID:-unknown}"; }
if grep -qi microsoft /proc/version 2>/dev/null; then
  bad "Windows/WSL detected — only NATIVE Ubuntu 24.04 is supported. Dual-boot it."
elif [[ "$os_id" == "ubuntu" && "$os_version" == "24.04" ]]; then
  ok "Ubuntu 24.04 — the supported platform"
else
  bad "${os_id} ${os_version} — NOT supported. This course requires Ubuntu 24.04 only."
fi
[[ -d /opt/ros/humble ]] && bad "ROS 2 Humble is installed — remove it: sudo apt purge -y 'ros-humble-*'"

# ── 2. Display server & GPU (the simulator GUI) ─────────────────────────────
echo; echo "[2/6] Display server & GPU (simulator GUI)"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  warn "No graphical session (headless/SSH). Run the sim from the laptop's own screen."
elif [[ "${XDG_SESSION_TYPE:-}" == "wayland" || -n "${WAYLAND_DISPLAY:-}" ]]; then
  ok "Wayland session — run_sim.sh auto-forces xcb (Gazebo GUI fix, handled)"
else
  ok "X11 session — native display for the Gazebo GUI"
fi
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
  ok "NVIDIA GPU — run_sim.sh applies PRIME offload"
else
  echo "  – No NVIDIA GPU — Gazebo renders on Intel/AMD or software GL (fine; may be slower)."
fi

# ── 3. Hardware ─────────────────────────────────────────────────────────────
echo; echo "[3/6] Hardware"
cores="$(nproc)"
ram_gb=$(( ($(awk '/^MemTotal:/ {print $2}' /proc/meminfo) + 524288) / 1048576 ))
disk_gb="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
(( cores >= 4 ))    && ok "CPU: ${cores} cores"      || warn "CPU: only ${cores} cores (4+ recommended)"
(( ram_gb >= 8 ))   && ok "RAM: ${ram_gb} GB"        || warn "RAM: ${ram_gb} GB (8+ GB recommended)"
(( disk_gb >= 10 )) && ok "Free disk: ${disk_gb} GB" || bad  "Free disk: ${disk_gb} GB (need 10+ GB)"

# ── 4. Internet ─────────────────────────────────────────────────────────────
echo; echo "[4/6] Internet"
curl -fsSL --max-time 10 https://packages.ros.org >/dev/null 2>&1 \
  || curl -fsSL --max-time 10 https://pypi.org >/dev/null 2>&1 \
  && ok "Internet reachable" || bad "Cannot reach packages.ros.org / pypi.org"

# ── 5. ROS 2 Jazzy stack ────────────────────────────────────────────────────
echo; echo "[5/6] ROS 2 Jazzy stack"
STACK_OK=0
if [[ -d /opt/ros/jazzy ]]; then
  set +u; source /opt/ros/jazzy/setup.bash 2>/dev/null; set -u
  command -v ros2 >/dev/null 2>&1 && ok "ROS 2 Jazzy (ros2 CLI works)" || bad "ROS 2 installed but 'ros2' not working"
  gzv="$(gz sim --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
  [[ "${gzv%%.*}" == "8" ]] && ok "Gazebo Harmonic (gz sim ${gzv})" || bad "Gazebo Harmonic missing (got: ${gzv:-none})"
  command -v colcon >/dev/null 2>&1 && ok "colcon build tool" || bad "colcon missing"
  command -v rosdep >/dev/null 2>&1 && ok "rosdep" || bad "rosdep missing"
  /usr/bin/python3 -c 'import numpy,cv_bridge; assert int(numpy.__version__.split(".")[0])<2' 2>/dev/null \
    && ok "cv_bridge + numpy<2 (camera labs)" || bad "cv_bridge/numpy broken — re-run install.sh"
  /usr/bin/python3 -c 'import flask' 2>/dev/null && ok "flask (phone teleop)" || bad "flask missing"
  command -v ros2 >/dev/null 2>&1 && [[ "${gzv%%.*}" == "8" ]] && STACK_OK=1
else
  bad "ROS 2 Jazzy not installed — run ./setup/install.sh"
fi

# ── 6. This workspace built ─────────────────────────────────────────────────
echo; echo "[6/6] Course workspace (raise_ros2_ws)"
if [[ -f "${WS_DIR}/install/setup.bash" ]]; then
  ok "Workspace built — ready to launch (./setup/run_sim.sh)"
else
  bad "Workspace not built — run ./setup/install.sh"
fi
# ML/VLA extras (optional; only later modules) — informational, never fails.
/usr/bin/python3 -c 'import ultralytics,openai' 2>/dev/null \
  && ok "ML extras present (YOLO + OpenAI) — perception/VLA labs ready" \
  || echo "  – ML/VLA extras not installed yet (fine for now) — later: ./setup/install.sh --full"

# ── Summary ─────────────────────────────────────────────────────────────────
echo; echo "════════════════════════════════════════════════════════"
echo "  FINAL SUMMARY   ✓ ${PASS} passed   ⚠ ${WARN} warnings   ✗ ${FAIL} failed"
if (( FAIL == 0 )) && (( STACK_OK )); then
  echo "  ✅ READY  (Ubuntu 24.04 + ROS 2 Jazzy)  →  ./setup/run_sim.sh"
else
  echo "  ❌ NOT READY YET  →  fix the ✗ items (usually: re-run ./setup/install.sh)"
fi
echo "════════════════════════════════════════════════════════"; echo
(( FAIL == 0 )) && (( STACK_OK ))
