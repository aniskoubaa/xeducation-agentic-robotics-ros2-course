# dataset_basics — reading a real robot dataset

Six short scripts that open the **reference dataset** the course ships
(`raiseschool/raise_ripeness_sort_ref`: 49 demonstrations of *"pick the red
tomato"*, 4228 frames at 10 Hz) and show, one lesson at a time, what is
actually inside it.

| file | lesson | what it shows | touches the dataset how? |
|---|---|---|---|
| `01_one_frame.py` | 3.2 | one frame = **image + state + action + task** | `LeRobotDataset` |
| `02_state_vs_action.py` | 3.2 / 3.4 | state is *measured*, action is *commanded* — the label | `LeRobotDataset` |
| `03_episodes.py` | 3.3 | frame → **episode** → dataset, and a filmstrip of one pick | `LeRobotDataset` |
| `04_on_disk.py` | 3.4 | the folder: `meta/` bookkeeping, `data/` frames | plain files |
| `05_the_metadata.py` | 3.4 | what each file in `meta/` is actually for | `json` + `pyarrow` |
| `06_the_parquet.py` | 3.4 | the table itself — and where the images really live | `pyarrow` |

The first three read the dataset the way training does. The last three read it as
plain files, so you can see there is nothing hidden.

## Run them

LeRobot needs numpy 2, which ROS's `cv_bridge` does not tolerate, so it lives in
its own virtualenv (see `raise2026_labs/day2/DATASET.md`):

```bash
ln -sfn ../../../datasets/raiseschool__raise_ripeness_sort_ref dataset      # 01-06 all read ./dataset
mkdir -p out

~/raise_venvs/lerobot/bin/python3 01_one_frame.py
~/raise_venvs/lerobot/bin/python3 02_state_vs_action.py
~/raise_venvs/lerobot/bin/python3 03_episodes.py
~/raise_venvs/lerobot/bin/python3 04_on_disk.py
~/raise_venvs/lerobot/bin/python3 05_the_metadata.py
~/raise_venvs/lerobot/bin/python3 06_the_parquet.py
```

`dataset/` is the symlink you just made and `out/` is what the scripts draw;
both are gitignored. The dataset itself (~36 MB) ships in this repo under
`datasets/`.

## Two things worth knowing

**The action is the command, not the pose.** `02` prints the gripper at the
grasp: commanded `0.500`, measured never above `0.412`, because the tomato is in
the way. Imitation copies your *intent* at each step — train on the measurement
and you teach the robot to stop halfway.

**There is no `videos/` folder here.** Slide 4 of Lesson 3.4 shows the general
LeRobot v3.0 layout, where camera frames become MP4 under `videos/`. This
dataset declares the wrist camera as dtype `image`, so LeRobot packs every frame
into `data/chunk-000/file-000.parquet` as PNG bytes instead — 636 MB of raw
pixels in a 35 MB file. `06` shows the column that holds them.
