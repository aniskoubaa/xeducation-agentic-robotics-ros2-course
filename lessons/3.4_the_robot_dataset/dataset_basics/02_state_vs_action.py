"""Lesson 3.2 / 3.4 - state is NOW, action is NEXT, and action is the LABEL.

The same seven numbers appear twice in every frame, and beginners mix them up:

    observation.state   where the arm IS       (measured)
    action              what you COMMANDED     (intent)  <- this is the label

The gripper proves the difference.  At the grasp we command it shut - and it
never gets there, because a tomato is in the way.  Train on the measurement
and you would teach the robot to stop halfway.

    python3 02_state_vs_action.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
DATASET = 'dataset'
GRIP = 6                             # the 7th number: 0.0 = open, 0.5 = closed
NOTES = {32: 'open, approaching', 33: 'the arm has not moved yet',
         40: 'closing onto the tomato', 60: 'stalled - the tomato is in the way'}

ds = LeRobotDataset(REPO_ID, root=DATASET)
start = ds.meta.episodes[0]['dataset_from_index']        # where episode 0 begins

print(' step    t       action[6] COMMANDED       state[6] MEASURED')
for step, note in NOTES.items():
    f = ds[start + step]
    cmd, meas = float(f['action'][GRIP]), float(f['observation.state'][GRIP])
    print(f'  {step:3d}  {float(f["timestamp"]):4.1f} s      {cmd:.3f}'
          f'  {"CLOSE" if cmd else "open ":5s}            {meas:.3f}   {note}')

peak = max(float(ds[start + s]['observation.state'][GRIP]) for s in range(76))
print(f'commanded 0.500, measured never above {peak:.3f}'
      '  ->  the label is the intent, not the pose')
