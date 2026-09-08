"""Lesson 3.6 - what the seven numbers in an action actually are.

WHAT  Ask the dataset itself for the NAME of every channel, then print one real
      action row with each number next to the joint it drives.

WHY   "action is 7 floats" is not yet readable.  A label you cannot name is a
      label you cannot check.  The names are not in a comment somewhere - they
      ship inside meta/info.json, so the dataset stays self-describing long
      after whoever recorded it has gone home.

    python3 02_the_seven_numbers.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
EPISODE, STEP = 1, 54                # the frame where this episode grasps

ds = LeRobotDataset(REPO_ID, root='dataset')
names = ds.meta.info['features']['action']['names']
frame = ds[ds.meta.episodes[EPISODE]['dataset_from_index'] + STEP]

spec = ds.meta.info['features']['action']
print(f'action {spec["shape"]} {spec["dtype"]} at t = {STEP / ds.fps:.1f} s of '
      f'episode {EPISODE} - meta/info.json names each channel '
      '(state uses the same seven):')
for i, (name, value) in enumerate(zip(names, frame['action'])):
    # The first six are the arm, in the order the controller expects them; the
    # seventh is the gripper, and it is the odd one out - see script 05.
    part = 'gripper  <- the odd one out' if i == 6 else f'arm joint {i + 1}'
    print(f'  action[{i}]  {value:+7.3f}   {name:<38s} {part}')
