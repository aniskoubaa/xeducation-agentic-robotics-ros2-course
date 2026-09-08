"""Lesson 3.6 - the same test, on all 49 episodes: does the robot reach the label?

WHAT  For every channel, compare the action at step t with the measured state a
      few steps later, across the whole dataset, and report the delay that fits
      best.  Read straight from the parquet - no images decoded, so it is quick.

WHY   One episode could be a fluke.  Over 4228 frames the pattern is flat: the
      six arm joints reach the commanded angle about ONE frame (0.1 s) later,
      and the gripper never reaches it at all.  So "the action is the command,
      not the pose" is not a style choice about how we recorded - it is the only
      description that fits the data.  The arm's label is a target it achieves;
      the gripper's label is an intent that contact overrules.

    python3 06_the_label_is_the_command.py
"""

import numpy as np
import pyarrow.parquet as pq

TABLE = 'dataset/data/chunk-000/file-000.parquet'
LAGS = range(10)                     # 0 .. 0.9 s at 10 Hz

# Only the three columns we need: the labels, the measurements, and the episode
# each row belongs to (so a shift never runs off one episode into the next).
table = pq.read_table(TABLE, columns=['action', 'observation.state', 'episode_index'])
action = np.stack(table['action'].to_numpy(zero_copy_only=False))
state = np.stack(table['observation.state'].to_numpy(zero_copy_only=False))
episode = table['episode_index'].to_numpy()
names = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3', 'GRIPPER']

print(f'{len(action)} frames, {len(set(episode))} episodes:  how far is the measured '
      'joint from the command it was given?')
for c, name in enumerate(names):
    gaps = []
    for k in LAGS:
        same = episode[:len(episode) - k] == episode[k:]      # stay inside one episode
        gaps.append(np.abs(action[:len(action) - k, c][same] - state[k:, c][same]).mean())
    best = int(np.argmin(gaps))
    print(f'  {name:<14s} same instant {gaps[0]:.3f} rad     best match {best / 10:+.1f} s later: '
          f'{gaps[best]:.3f} rad = {np.degrees(gaps[best]):.1f} deg')

# The arm's numbers are small because the arm gets there.  The gripper's are not
# a delay at all - so state the gripper case exactly, with no averaging to hide
# behind: does the measured joint EVER sit on a value we commanded?
commanded = sorted({round(float(v), 3) for v in action[:, 6]})
on_target = sum(min(abs(float(m) - c) for c in commanded) < 0.02 for m in state[:, 6])
print(f'the gripper is only ever commanded {commanded[0]:.1f} or {commanded[-1]:.1f}, and measures '
      f'{state[:, 6].min():.3f}..{state[:, 6].max():.3f}: on target in {on_target} of {len(state)} frames.')
