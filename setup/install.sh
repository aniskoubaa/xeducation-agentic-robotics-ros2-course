#!/usr/bin/env bash
# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
# Agentic Robotics with ROS 2 — one-command machine setup.
#
# Installs everything needed to run the emulated RAISE-Bot, then BUILDS the
# workspace shipped in this repo (raise_ros2_ws/):
#   • ROS 2 Jazzy desktop + dev tools
#   • Gazebo Harmonic bridge (ros_gz)
#   • apt lab dependencies (Nav2, xacro, cv_bridge, OpenCV, Flask, colcon, rosdep)
#   • builds raise_ros2_ws with colcon and sources it in new shells
#   • (with --full) the ML/VLA extras used by later modules:
#        numpy<2 + torch/torchvision/ultralytics/openai, and a LeRobot venv
#
# ── PLATFORM POLICY ─────────────────────────────────────────────────────────
#   Ubuntu 24.04 + ROS 2 Jazzy is the ONLY supported platform. No 22.04, no
#   WSL, no macOS, no Docker. Not 24.04 → the script refuses (dual-boot 24.04).
#
# Usage:  ./setup/install.sh          # core: ROS + Gazebo + build the workspace
#         ./setup/install.sh --full   # + YOLO/OpenAI + LeRobot venv (~4 GB, later modules)
#         (safe to re-run — every step is idempotent)
set -euo pipefail

DISTRO=jazzy
FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

# ── 0. Ubuntu 24.04 gate ────────────────────────────────────────────────────
refuse() {
  cat >&2 <<EOF

╔══════════════════════════════════════════════════════════════════════════╗
║  This course supports ONE platform: Ubuntu 24.04 LTS + ROS 2 Jazzy.       ║
╚══════════════════════════════════════════════════════════════════════════╝

  Detected: $1

  Install Ubuntu 24.04 LTS (Desktop) — dual-boot alongside Windows is fine —
  boot into it, and run this script there.
  Not supported: other Ubuntu versions, other Linux, Windows/WSL, macOS, Docker.
EOF
  exit 1
}

[[ "$(uname -s)" == "Linux" ]] || refuse "non-Linux OS ($(uname -s))"

# Anaconda breaks ROS tooling (no catkin_pkg) — re-exec with it stripped.
if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "[install] Anaconda env '${CONDA_DEFAULT_ENV}' active — re-exec'ing without it."
  exec env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL \
    PATH="$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | paste -sd:)" \
    bash "$0" "$@"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WS_DIR="${REPO_ROOT}/raise_ros2_ws"

os_id=unknown; os_version=unknown; os_codename=unknown
if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  os_id="${ID:-unknown}"; os_version="${VERSION_ID:-unknown}"; os_codename="${VERSION_CODENAME:-unknown}"
fi
grep -qi microsoft /proc/version 2>/dev/null && refuse "Windows/WSL (only NATIVE Ubuntu 24.04 is supported)"
[[ "$os_id" == "ubuntu" ]]     || refuse "${os_id} ${os_version} (not Ubuntu)"
[[ "$os_version" == "24.04" ]] || refuse "Ubuntu ${os_version} (only 24.04 is supported)"

# ── 1. Machine check ────────────────────────────────────────────────────────
cpu_cores="$(nproc)"
ram_gb=$(( ($(awk '/^MemTotal:/ {print $2}' /proc/meminfo) + 524288) / 1048576 ))
disk_free_gb="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
cat <<EOF

Agentic Robotics with ROS 2 — machine check
  OS:        Ubuntu ${os_version} (${os_codename})  → ROS 2 ${DISTRO^}  ✓ supported
  CPU:       ${cpu_cores} cores    RAM: ${ram_gb} GB    Free disk: ${disk_free_gb} GB on /
  Mode:      $([[ $FULL == 1 ]] && echo 'FULL (+ ML/VLA extras)' || echo 'core (ROS + Gazebo + build)')
EOF
(( ram_gb < 8 ))        && echo "  ⚠ < 8 GB RAM — the simulator will be slow. Close other apps during labs."
(( disk_free_gb < 25 )) && echo "  ⚠ < 25 GB free disk — free up space."

# ── 2. Remove ROS 2 Humble if present (unsupported) ─────────────────────────
if [[ -d /opt/ros/humble ]]; then
  echo
  echo "⚠ ROS 2 Humble found at /opt/ros/humble — it must be removed (Jazzy only)."
  echo "    sudo apt purge -y 'ros-humble-*' && sudo apt autoremove -y"
  if [[ -t 0 ]]; then
    read -r -p "Remove ROS 2 Humble now? [Y/n]: " ans || true
    [[ ! "${ans:-Y}" =~ ^[Nn] ]] && { sudo apt purge -y 'ros-humble-*' && sudo apt autoremove -y; }
  fi
  grep -q '/opt/ros/humble/setup.bash' "${HOME}/.bashrc" 2>/dev/null \
    && sed -i '\|/opt/ros/humble/setup.bash|d' "${HOME}/.bashrc"
fi

# ── 3. ROS 2 Jazzy (idempotent) ─────────────────────────────────────────────
if [[ ! -d "/opt/ros/${DISTRO}" ]]; then
  echo "[install] Installing ROS 2 ${DISTRO^} (10–30 min)…"
  sudo apt update
  sudo apt install -y software-properties-common curl gnupg lsb-release locales
  sudo locale-gen en_US en_US.UTF-8
  sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
  sudo add-apt-repository -y universe
  sudo install -d -m 0755 /usr/share/keyrings
  sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${os_codename} main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
  sudo apt update
  sudo apt install -y "ros-${DISTRO}-desktop" ros-dev-tools
  echo "[install] ✓ ROS 2 ${DISTRO^} installed."
else
  echo "[install] ✓ ROS 2 ${DISTRO^} already installed — skipping."
fi

# ── 4. Gazebo Harmonic bridge + lab apt deps ────────────────────────────────
echo "[install] Installing Gazebo Harmonic bridge + lab dependencies (apt)…"
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-gz ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-sim \
  "ros-${DISTRO}-nav2-bringup" "ros-${DISTRO}-xacro" \
  "ros-${DISTRO}-robot-state-publisher" "ros-${DISTRO}-cv-bridge" \
  python3-opencv python3-flask python3-colcon-common-extensions \
  python3-rosdep python3-venv python3-pip mesa-utils git

# Optional robot-description packages (best-effort; rosdep fills any gaps).
for p in "ros-${DISTRO}-clearpath-simulator" "ros-${DISTRO}-ur-description" \
         "ros-${DISTRO}-realsense2-description" "ros-${DISTRO}-robotiq-description"; do
  sudo apt install -y "$p" >/dev/null 2>&1 && echo "[install] ✓ $p" \
    || echo "[install] ⚠ optional package unavailable, skipping: $p"
done

# ── 5. (--full only) ML libs for perception labs (numpy<2 in the SAME resolve) ─
if (( FULL )); then
  echo "[install] --full: installing Python ML libraries (pip, ~1 GB)…"
  PIP_FLAGS=(--user)
  /usr/bin/python3 -m pip install --help 2>/dev/null | grep -q -- '--break-system-packages' \
    && PIP_FLAGS+=(--break-system-packages)
  for attempt in 1 2 3; do
    /usr/bin/python3 -m pip install "${PIP_FLAGS[@]}" --retries 5 --timeout 120 \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "numpy<2" torch torchvision ultralytics openai && break
    echo "[install] pip attempt ${attempt} failed — retrying in 10 s…"; sleep 10
    [[ "$attempt" == "3" ]] && { echo "[install] ✗ pip failed 3×. Re-run with --full." >&2; exit 1; }
  done

  echo "[install] --full: installing the LeRobot AI toolkit venv (~3 GB)…"
  VENV="${HOME}/raise_venvs/lerobot"
  if ! "${VENV}/bin/python3" -c 'import lerobot' 2>/dev/null; then
    /usr/bin/python3 -m venv --system-site-packages "${VENV}"
    "${VENV}/bin/python3" -m pip install --upgrade pip >/dev/null 2>&1 || true
    for attempt in 1 2 3; do
      "${VENV}/bin/python3" -m pip install --timeout 120 --retries 10 "lerobot[smolvla]" \
        && { echo "[install] ✓ LeRobot installed"; break; }
      echo "[install] LeRobot attempt ${attempt} failed — retrying…"; sleep 10
    done
  fi
fi

# ── 6. rosdep ───────────────────────────────────────────────────────────────
[[ -f /etc/ros/rosdep/sources.list.d/20-default.list ]] || sudo rosdep init
rosdep update

# ── 7. ufw multicast (gz-transport discovery; GUI hangs without it) ─────────
if command -v ufw >/dev/null 2>&1 && sudo ufw status 2>/dev/null | grep -q "Status: active"; then
  sudo ufw allow in proto udp to 224.0.0.0/4   >/dev/null || true
  sudo ufw allow in proto udp from 224.0.0.0/4 >/dev/null || true
fi

# ── 8. Disable Gazebo first-run Quick Start dialog (hangs on Fuel calls) ────
GZ_USER_CFG="${HOME}/.gz/sim/8/gui.config"
mkdir -p "$(dirname "${GZ_USER_CFG}")"
if [[ -f "${GZ_USER_CFG}" ]]; then
  sed -i 's|<dialog name="quick_start" show_again="true"/>|<dialog name="quick_start" show_again="false"/>|' "${GZ_USER_CFG}"
else
  printf '<?xml version="1.0"?>\n<dialog name="quick_start" show_again="false"/>\n' > "${GZ_USER_CFG}"
fi

# ── 9. Build the workspace shipped in this repo ─────────────────────────────
echo "[install] Building the workspace (colcon)…"
cd "${WS_DIR}"
set +u; source "/opt/ros/${DISTRO}/setup.bash"; set -u
rosdep install --from-paths src --ignore-src -r -y || true
colcon build --symlink-install --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
echo "[install] ✓ workspace built at ${WS_DIR}/install"

# ── 10. Auto-source ROS 2 + workspace + aliases in every new terminal ───────
# One managed block sources setup/aliases.sh, which loads ROS 2, this workspace,
# any API keys, and installs the raise-sim / raise-stop / raise-check helpers.
BASHRC="${HOME}/.bashrc"
if ! grep -q ">>> agentic-robotics-ros2 >>>" "${BASHRC}" 2>/dev/null; then
  {
    echo ""
    echo "# >>> agentic-robotics-ros2 >>>"
    echo "source ${SCRIPT_DIR}/aliases.sh"
    echo "# <<< agentic-robotics-ros2 <<<"
  } >> "${BASHRC}"
  echo "[install] Added ROS 2 + workspace + aliases to ~/.bashrc"
fi

cat <<EOF

✓ Setup complete  (Ubuntu ${os_version} + ROS 2 ${DISTRO^}).

  Open a NEW terminal, then just type:
      raise-sim        # launch the greenhouse simulation
      raise-check      # confirm your machine is ready
      raise-stop       # stop the simulation
$([[ $FULL == 0 ]] && echo "  Perception/VLA labs later? re-run with:  ./setup/install.sh --full")
EOF

# ── 11. Readiness check ─────────────────────────────────────────────────────
[[ -f "${SCRIPT_DIR}/check_setup.sh" ]] && { echo; bash "${SCRIPT_DIR}/check_setup.sh" || true; }
