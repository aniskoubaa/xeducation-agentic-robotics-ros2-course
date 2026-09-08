# Getting started

From a blank Ubuntu machine to a robot driving through a greenhouse. Do this
**once, before the course** — it takes about half an hour, most of it download.

---

## 0. Before anything: the platform

| | |
|---|---|
| **Operating system** | Ubuntu **24.04 LTS** Desktop, installed natively |
| **ROS 2** | Jazzy — installed for you by the script |
| **Gazebo** | Harmonic — installed for you by the script |
| **RAM** | 8 GB minimum |
| **Free disk** | 10 GB for the core install, ~15 GB with `--full` |
| **GPU** | Helpful, not required — there is a lighter world for machines without one |

**Only native Ubuntu 24.04 is supported.** Not WSL, not macOS, not a virtual
machine, not Docker. `install.sh` checks and refuses to run anywhere else, and
that refusal is deliberate: this is a GPU-accelerated 3D simulator, and every
one of those layers is a known way for it to fail. Dual-booting 24.04 alongside
Windows is fine and is what most students do.

---

## 1. Get the code

```bash
git clone https://github.com/aniskoubaa/xeducation-agentic-robotics-ros2-course.git
cd xeducation-agentic-robotics-ros2-course
```

The course code is now on your machine, ready to build.

---

## 2. Install everything

```bash
./setup/install.sh
```

It asks for your password, because it installs system software. Then, with no
further input from you, it:

1. checks you are on Ubuntu 24.04, and stops if you are not;
2. installs **ROS 2 Jazzy** (desktop + dev tools);
3. installs **Gazebo Harmonic** and the `ros_gz` bridge;
4. installs the course apt dependencies (Nav2, xacro, `cv_bridge`, OpenCV, Flask, colcon, rosdep);
5. resolves the workspace dependencies with `rosdep`;
6. **builds `raise_ros2_ws/`** with colcon;
7. appends a small managed block to your `~/.bashrc` so every new terminal has ROS 2, this workspace and the course aliases already sourced.

It is idempotent — if a step fails on a flaky network, just run it again.

### The `--full` variant

Modules 0 through 2 need nothing more than the above. The vision and VLA
lessons of module 3 need the ML stack, which is another ~4 GB:

```bash
./setup/install.sh --full     # + numpy<2, torch, torchvision, ultralytics, openai
                              # + a LeRobot virtualenv at ~/raise_venvs/lerobot
```

You can run this later, when you reach module 3. Running it now is fine too.

### Open a new terminal

The aliases are installed into `~/.bashrc`, so they exist in **new** terminals.
Close the one you installed from and open a fresh one.

---

## 3. Check the machine

```bash
./setup/check_setup.sh
```

Six checks — OS, display server and GPU, RAM and disk, internet, the ROS 2
stack, and whether the workspace is built — ending in either

```
✅ READY  (Ubuntu 24.04 + ROS 2 Jazzy)  →  ./setup/run_sim.sh
```

or a list of `✗` items to fix. Almost every `✗` is cured by re-running
`./setup/install.sh`. Run this any time something stops working; it is also
available as `raise-check`.

---

## 4. First run

Two terminals.

```bash
# terminal 1 — the robot and its world
raise-sim
```

Gazebo opens on a greenhouse: tomato rows, the Husky robot with a UR5e arm on
its back. Leave it running; it is the world everything else talks to.

```bash
# terminal 2 — drive it yourself
raise-teleop
```

`w`/`s` drive, `a`/`d` turn, SPACE stops. Or call one of the robot's skills
directly, without writing a line of code:

```bash
ros2 service call /nav_to_tomato_row_1 std_srvs/srv/Trigger
```

If the robot drives and the service comes back with success, **you are ready**.

```bash
raise-stop     # shut the simulation down cleanly when you are done
```

### If Gazebo is slow

On a laptop with no discrete GPU, use the lighter world:

```bash
./setup/run_sim.sh world:=greenhouse_2026_lite.sdf
```

---

## 5. The rhythm of every hands-on

Every exercise in the course has the same shape, so learn it once:

```bash
raise-sim                                     # terminal 1: the robot must be running
ros2 run raise2026_labs 01_drive_forward.py   # terminal 2: one small script against it
```

Start the simulation, then run one small program against it. Most lab scripts
start the skill server they need on their own, so step 2 is usually all you
type. When something goes wrong, you have two safety commands: `raise-check`
says what is missing, `raise-stop` gives you a clean slate.

---

## 6. API keys (module 2 onward)

The robot, its skills, and all of ROS 2 run entirely on your machine. Nothing
leaves it and no key is needed for modules 0 and 1.

From module 2 the agent calls an LLM, so you need one key of your own —
either OpenAI or Google Gemini, depending on which lesson script you run:

```bash
cp .env.example raise_ros2_ws/src/.env
nano raise_ros2_ws/src/.env        # paste your key in
```

Every new terminal loads that file. It is gitignored — **never commit a key**,
and never paste one into a source file.

---

## When something breaks

| Symptom | What to do |
|---|---|
| `raise-sim` not found | Open a **new** terminal — the aliases land in `~/.bashrc` |
| Gazebo opens a blank window | You are on Wayland; `run_sim.sh` already forces XWayland — launch through it, not `gz sim` directly |
| `gz sim` symbol errors | You are in a VS Code snap terminal; use a normal terminal |
| `No module named catkin_pkg` | Anaconda is shadowing system Python; the scripts strip it, so run them rather than the commands by hand |
| Ogre2 hangs on a hybrid laptop | `run_sim.sh` applies NVIDIA PRIME offload — again, launch through it |
| Anything else | `./setup/check_setup.sh`, then [open an issue](https://github.com/aniskoubaa/xeducation-agentic-robotics-ros2-course/issues) with its output |

---

Now go and watch
[lesson 0.4 — Your First Run](https://learn.xeducation.ai/courses/agentic-robotics).
