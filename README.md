# Agentic Robotics with ROS 2 — course code

Everything you run in **[Agentic Robotics with ROS 2: From LLM Tool-Calling to
Vision-Language-Action Models](https://learn.xeducation.ai/courses/agentic-robotics)**
— the simulated robot, its skills, the agent scripts, and the demo code shown in
the lectures. One repository, three commands, and the greenhouse is running on
your machine.

> **Watch the course on [learn.xeducation.ai](https://learn.xeducation.ai/courses/agentic-robotics).**
> This repository is the code half; the videos, quizzes and certificate live there.

---

## Three commands

```bash
git clone https://github.com/aniskoubaa/xeducation-agentic-robotics-ros2-course.git
cd xeducation-agentic-robotics-ros2-course
./setup/install.sh        # ROS 2 Jazzy + Gazebo Harmonic + deps, then builds the workspace
./setup/run_sim.sh        # launches the greenhouse simulation
```

Check your machine at any time:

```bash
./setup/check_setup.sh    # prints a green READY when everything is in place
```

**Platform:** native **Ubuntu 24.04 LTS**. That is the only supported setup — no
WSL, no macOS, no virtual machine, no Docker. This is a GPU-accelerated 3D
simulator; the layers those options add are exactly the ones that break it. If
you are not on 24.04, dual-boot it before you start.

Full walkthrough, including what to do when a step fails:
**[GETTING_STARTED.md](GETTING_STARTED.md)**.

---

## What is in here

| Folder | What it holds |
|---|---|
| `setup/` | The installer, the launcher, the readiness check, and the shell aliases |
| `raise_ros2_ws/` | The ROS 2 workspace — seven packages, the robot and everything it can do |
| `lessons/` | The demo code shown on screen in the lectures, one folder per lesson |

### The workspace, package by package

`raise_ros2_ws/src/` holds seven packages, and they follow the course itself:
from the robot and its world, to its skills, to the intelligence that uses them.

| Package | Role |
|---|---|
| `raise2026_description` | The robot: Husky base, UR5e arm, gripper, cameras (URDF/xacro) |
| `raise2026_worlds` | The greenhouse: Gazebo worlds and plant meshes |
| `raise2026_bringup` | Launches world + robot + cameras together |
| `raise2026_tools` | The robot's **skills** — the ROS 2 services an LLM agent calls |
| `raise2026_teleop` | Drive it yourself: keyboard, joystick, or phone |
| `raise2026_labs` | The numbered hands-on scripts |
| `raise2026_demos` | One demo node per lecture |

`raise2026_tools` is the heart of the system: every skill the agent can invoke —
navigate, grasp, inspect, detect — is a ROS 2 service in there.

### The lesson demos

`lessons/` mirrors the course numbering, so the folder name is the lesson name.
Each one is a handful of short scripts with a README that says what each shows:

| Lesson | Folder |
|---|---|
| 2.0 What is an LLM? | `lessons/2.0_what_is_an_llm/llm_basics/` |
| 2.1 Agentic AI | `lessons/2.1_agentic_ai/agent_basics/` |
| 2.2 ReAct prompting | `lessons/2.2_react_prompting/react_basics/` |
| 2.4 MCP tools | `lessons/2.4_mcp_tools/mcp_basics/` |
| 2.5 The agent loop | `lessons/2.5_agent_loop/agent_basics/` |
| 3.4 The robot dataset | `lessons/3.4_the_robot_dataset/dataset_basics/` |
| 3.5 Teleoperation | `lessons/3.5_teleoperation/teleop_demo/` |
| 3.6 Reading action labels | `lessons/3.6_reading_action_labels/labels_demo/` |
| 3.7 How a model moves | `lessons/3.7_how_a_model_moves/action_words_demo/` |
| 3.8 Action generation | `lessons/3.8_action_generation/action_chunks_demo/` |
| 3.11 Meet SmolVLA | `lessons/3.11_meet_smolvla/` (three demos) |
| 3.12 Evaluating a policy | `lessons/3.12_evaluating_a_policy/eval_demo/` |

---

## The rhythm of every hands-on

It is the same every time, so learn it once:

```bash
raise-sim                                   # terminal 1 — leave the simulation running
ros2 run raise2026_labs 01_drive_forward.py # terminal 2 — run one small script against it
```

Start the simulation, then run one small program against it. If something goes
wrong, `raise-check` tells you exactly what is missing and `raise-stop` gives you
a clean slate.

### The aliases the installer gives you

Every **new** terminal gets these:

| Type this | and it… |
|---|---|
| `raise-sim` | launches the greenhouse simulation |
| `raise-teleop` | drives the robot from the keyboard (`w/s` `a/d`, SPACE) |
| `raise-stop` | stops the simulation cleanly |
| `raise-check` | verifies your machine is ready |

Plus shortcuts for the skill servers (`gripper_server`, `navigation_server`,
`detector_server`, …).

---

## API keys

Modules 0 and 1 need no key at all — the robot, its skills and all of ROS 2 run
entirely on your machine. The agent lessons in module 2 call an LLM, so from
there on you need one key of your own:

```bash
cp .env.example raise_ros2_ws/src/.env   # then put your key in it
```

Every new terminal loads it. `raise_ros2_ws/src/.env` is gitignored — **never
commit a key**, and never paste one into a source file.

---

## Getting help

Something not working? Run `./setup/check_setup.sh` first — it names the problem
in most cases. If that does not settle it, open an
[issue](https://github.com/aniskoubaa/xeducation-agentic-robotics-ros2-course/issues)
with the output of that command pasted in.

---

## License

Code released under the MIT License — see [LICENSE.md](LICENSE.md). The course
videos, slides and narration are **not** covered by it; they remain
© ScaleX Innovation, all rights reserved.

Built by [Anis Koubaa](https://github.com/aniskoubaa) for
[XEducation](https://learn.xeducation.ai).
