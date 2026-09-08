# Lesson 3.5 — teleoperation, on the live robot

Where the episodes come from. Everything here runs against the **running sim**:
the arm really moves, and the dataset that appears at the end is the same
LeRobot folder we opened in lesson 3.4.

| what | file | it shows |
|---|---|---|
| the lab's own scripts | `starter/` | teleop, the stream check, the recorder, the auto-demonstrator |
| drive it by hand | `drive.py` | presses real teleop keys — one key, one joint target |
| what came out | `check_dataset.py` | the fresh recording next to the reference set, plus a filmstrip |

## Run it

Three things must be up — the sim parked at the plant row, and the grasp server:

```bash
raise-sim x:=-2.0 y:=2.15 yaw:=1.5708      # terminal 1
ros2 run raise2026_tools grasp_server      # terminal 2
```

Then, from this folder:

```bash
ros2 run raise2026_labs 02_read_streams.py                 # the two streams, live
python3 drive.py                                           # drive the arm by hand
ros2 run raise2026_labs 05_auto_demonstrate.py \
    --episodes 3 --team demo --data-root out               # record, hands-free
python3 check_dataset.py                                   # what we just made
```

`python3` for `check_dataset.py` must be the LeRobot venv
(`~/raise_venvs/lerobot/bin/python3`); the lab's `ros2 run` scripts re-exec
themselves into it automatically.

## Worth knowing

- **The park pose matters.** `sim_poses.py` is tuned for `x:=-2.0 y:=2.15
  yaw:=1.5708`. Launch the sim anywhere else and the arm still records fine (the
  demonstrator spawns each tomato where the tool actually lands), but the wrist
  camera no longer looks at the plant row.
- **The action is the command, not the pose.** `03_record.py` reads `state` from
  `/joint_states` and `action` from the `/…/cmd` topics your teleop publishes —
  the same distinction lesson 3.4 measured on the reference set.
- **`out/` and `ref` are gitignored** — `ref` is a symlink to the reference
  dataset that ships with the Day-2 labs.
