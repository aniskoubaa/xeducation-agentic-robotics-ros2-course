# Setup — get the RAISE-Bot running

**Platform:** native **Ubuntu 24.04 LTS** + **ROS 2 Jazzy** + **Gazebo Harmonic** — the only supported setup. No WSL, no macOS, no VM, no Docker. Not on 24.04? Dual-boot it.

## Three commands

```bash
git clone https://github.com/aniskoubaa/agentic-robotics-ros2.git
cd agentic-robotics-ros2
./setup/install.sh        # installs ROS 2 + Gazebo + deps, builds raise_ros2_ws/  (asks for sudo)
./setup/run_sim.sh        # launches the greenhouse simulation
```

Verify any time:

```bash
./setup/check_setup.sh    # prints a green READY when your machine is good
```

## Scripts

| Script | What it does |
|---|---|
| `install.sh` | Ubuntu-24.04 gate → ROS 2 Jazzy → Gazebo Harmonic (`ros_gz`) → lab apt deps → `rosdep` → `colcon build` the workspace → auto-source in new shells. Idempotent (safe to re-run). |
| `install.sh --full` | Also installs the ML/VLA extras used by later modules: `numpy<2` + `torch/torchvision/ultralytics/openai`, and a `~/raise_venvs/lerobot` venv. ~4 GB. |
| `run_sim.sh` | Launches `raise2026_bringup sim.launch.py` with the environment fixes students hit: Wayland→`xcb`, NVIDIA PRIME offload, snap/anaconda path stripping. |
| `check_setup.sh` | Readiness report (OS, GPU, RAM/disk, internet, ROS 2 stack, workspace built). |

## Aliases (auto-installed to `~/.bashrc`)

`install.sh` adds a managed block sourcing `aliases.sh`, so every **new** terminal has:

| Type this | and it… |
|---|---|
| `raise-sim` | launches the greenhouse simulation |
| `raise-teleop` | drives the robot from the keyboard (`w/s a/d`, SPACE) |
| `raise-stop` | stops the simulation cleanly |
| `raise-check` | verifies your machine is ready |

Plus skill-server shortcuts (`gripper_server`, `navigation_server`, `detector_server`, …) and automatic sourcing of ROS 2 + this workspace + your `raise_ros2_ws/src/.env` keys.

## First run

```bash
raise-sim                                          # terminal 1: bring up the robot
raise-teleop                                       # terminal 2: w/s drive, a/d turn, SPACE stop
ros2 service call /nav_to_tomato_row_1 std_srvs/srv/Trigger   # or call a skill by hand
raise-stop                                         # when done
```

If the robot drives and the service returns success, you're ready. Do this **once, before the course**.

> Only the vision-language and agent labs need an `OPENAI_API_KEY` (export it in `~/.bashrc` or put it in `raise_ros2_ws/src/.env`). Not needed to get started.

*Adapted from the proven RAISE 2026 `prepare_machine.sh` / `raise-sim` — every hard-won fix (numpy<2 pinning, ufw multicast, Gazebo first-run dialog, Anaconda stripping) is preserved.*
