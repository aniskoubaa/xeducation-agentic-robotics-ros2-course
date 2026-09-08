# `raise_ros2_ws/` — the course ROS 2 workspace

The colcon workspace shipped with the course. Contains only the **`raise2026_*` packages** we author — third-party dependencies (Husky, UR description, MoveIt 2, Nav2, gazebo_ros) come in via `rosdep` at install time, not vendored here.

## Packages

| Package                 | Build type     | Role                                                          |
| ----------------------- | -------------- | ------------------------------------------------------------- |
| `raise2026_description` | `ament_cmake`  | URDF/xacro for Husky A200 + UR5e + Robotiq 2F-85 + RealSense  |
| `raise2026_worlds`      | `ament_cmake`  | Gazebo Harmonic `.sdf` greenhouse worlds + plant meshes       |
| `raise2026_bringup`     | `ament_python` | Launch files: world + robot + cameras + nav stack             |
| `raise2026_tools`       | `ament_python` | ROS 2 services the LLM agent calls (`move_to_pose`, etc.)     |
| `raise2026_demos`       | `ament_python` | One demo node per lecture (≤80 lines each)                    |

## Build

You do not normally build this by hand. `../setup/install.sh` installs ROS 2
Jazzy + Gazebo Harmonic + the dependencies and then builds this workspace for
you on Ubuntu 24.04. To build it manually inside an environment that is already
set up:

```bash
cd raise_ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Check it afterwards with `../setup/check_setup.sh`.

## What is tracked

Only `src/`. `build/`, `install/` and `log/` are gitignored (see `.gitignore`).
