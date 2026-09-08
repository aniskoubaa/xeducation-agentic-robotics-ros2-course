"""Lesson 3.6 - the grasp moment: the command jumps, the joint cannot follow.

WHAT  Zoom in on the gripper channel around the grasp and print, frame by frame,
      the COMMANDED value next to the MEASURED one - then draw the same thing.

WHY   Slide 4, right panel.  The command is a step: 0.0 -> 0.5, one frame.  The
      measured joint is not: it takes a moment to start, closes slowly, and then
      stops early - the tomato is in the way.  That gap is not an error in the
      data, it IS the contact.  A model trained on the measurement would learn
      to stop halfway; trained on the command, it learns to squeeze and hold.

    python3 05_the_grasp_moment.py
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
EPISODE, GRIP = 1, 6

ds = LeRobotDataset(REPO_ID, root='dataset')
ep = ds.meta.episodes[EPISODE]
start, length = ep['dataset_from_index'], ep['length']
frames = [ds[start + t] for t in range(length)]
cmd = np.array([float(f['action'][GRIP]) for f in frames])
meas = np.array([float(f['observation.state'][GRIP]) for f in frames])
time = np.arange(length) / ds.fps

close = int(np.argmax(cmd > 0.25))
release = close + int(np.argmax(cmd[close:] < 0.25))
peak = int(np.argmax(meas))
moved = close + int(np.argmax(meas[close:] > meas[close] + 0.01))   # first real response
stall = close + int(np.argmax(meas[close:] > meas[peak] - 0.01))    # where it gives up

print('  t        commanded            measured')
for t, note in ((close - 1, 'open, the tool is on the tomato'), (close, 'the COMMAND steps up'),
                (moved, f'the joint starts to move, {(moved - close) / ds.fps:.1f} s later'),
                (close + 10, 'still closing'), (stall, 'stalled - the tomato is in the way'),
                (release, 'released')):
    print(f' {t / ds.fps:4.1f} s     {cmd[t]:.3f}  {"CLOSE" if cmd[t] > 0.25 else "open ":5s}'
          f'      {meas[t]:.3f}   {note}')
print(f'commanded {cmd[close]:.3f}, measured never past {meas[peak]:.3f}: a gap of '
      f'{cmd[close] - meas[peak]:.3f} rad that never closes.  The label is the intent.')

fig, ax = plt.subplots(figsize=(18, 3.4), dpi=100)
ax.plot(time, cmd, lw=2.2, color='crimson', label='gripper COMMAND  (action[6] - the label)')
ax.plot(time, meas, lw=2.2, ls='--', color='darkorange',
        label='gripper MEASURED (observation.state[6] - stalls on the tomato)')
ax.axvspan(close / ds.fps, stall / ds.fps, color='crimson', alpha=0.08)
ax.annotate('command steps up', (close / ds.fps, 0.5), xytext=(-4, -14),
            textcoords='offset points', ha='right', fontsize=10, color='crimson')
ax.annotate(f'stalls at {meas[peak]:.3f}, never {cmd[close]:.1f}', (stall / ds.fps, meas[stall]),
            xytext=(10, -22), textcoords='offset points', fontsize=10, color='darkorange')
ax.set_xlabel('time [s]'); ax.set_ylabel('gripper [rad]')
ax.set_xlim(0, time[-1]); ax.grid(alpha=0.25); ax.legend(loc='upper left', fontsize=10)
fig.tight_layout()
fig.savefig('out/grasp_moment.png')
print('wrote out/grasp_moment.png')
