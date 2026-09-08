# labels_demo — reading the action labels

Lesson 3.4 asked *what is in the dataset*. This asks the next question: **what do
the numbers say?** Six scripts open the reference set
(`raiseschool/raise_ripeness_sort_ref`: 49 picks, 4228 frames at 10 Hz) and read
one episode of it the way a person would read a chart.

| file | it shows | reads the dataset how? |
|---|---|---|
| `01_what_the_robot_saw.py` | twelve wrist frames of one episode, in time order | `LeRobotDataset` |
| `02_the_seven_numbers.py` | what each of the 7 action channels **is**, named by the dataset itself | `LeRobotDataset` |
| `03_plot_the_curve.py` | the label drawn: what it saw above what it did, one time axis | `LeRobotDataset` |
| `04_read_the_curve.py` | the same curve read back as a sentence, phase by phase | `LeRobotDataset` |
| `05_the_grasp_moment.py` | the grasp: the command steps, the joint stalls on the tomato | `LeRobotDataset` |
| `06_the_label_is_the_command.py` | the same test over all 49 episodes | `pyarrow` (no images decoded) |

Everything is read from **episode 1** — the episode slide 4 of the lesson plots:
a full scan across the row, then a pick, 97 frames over 9.6 s.

## Run them

LeRobot needs numpy 2, which ROS's `cv_bridge` does not tolerate, so it lives in
its own virtualenv (see `raise2026_labs/day2/DATASET.md`):

```bash
ln -sfn ../../../datasets/raiseschool__raise_ripeness_sort_ref dataset      # 01-06 all read ./dataset
mkdir -p out

~/raise_venvs/lerobot/bin/python3 01_what_the_robot_saw.py
~/raise_venvs/lerobot/bin/python3 02_the_seven_numbers.py
~/raise_venvs/lerobot/bin/python3 03_plot_the_curve.py
~/raise_venvs/lerobot/bin/python3 04_read_the_curve.py
~/raise_venvs/lerobot/bin/python3 05_the_grasp_moment.py
~/raise_venvs/lerobot/bin/python3 06_the_label_is_the_command.py
```

`dataset/` and `out/` are gitignored.

## Three things worth knowing

**The gripper decides what a motion means.** `04` groups the frames where the
same joints are moving and names each group. The *same* shoulder sweep is the
**scan** with the gripper open and the **carry** with it closed — so the phase
label falls out of the whole state vector, not out of one channel. That is a
small, concrete reason a policy is given the entire observation.

**The measured gripper never reaches a commanded value — not once.** `06`
checks all 4228 frames: the command is only ever `0.0` or `0.5`, the measurement
only ever lands between `0.056` and `0.424`, and it is within 0.02 rad of the
command in **0** frames. The six arm joints, by contrast, arrive about one frame
(0.1 s) after they are told to. Same seven-number vector, two different kinds of
label: a target the arm reaches, and an intent that contact overrules.

**`wrist_3` is the noisy one.** Its gap (0.053 rad) does not shrink at any lag,
but that is jitter, not contact: the joint barely travels (0.22 rad in total)
and the measurement wobbles either side of the setpoint. Do not read its row in
`06` as a second grasp story.
