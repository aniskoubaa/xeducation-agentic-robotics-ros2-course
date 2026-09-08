"""Lesson 3.6 - the label, drawn: what it saw over what it did.

WHAT  Plot all seven ACTION channels of one episode against time, with the same
      episode's wrist frames laid along the top on the same clock.

WHY   Slide 4 shows these curves.  They are the label - the exact numbers the
      VLA is trained to output for the images above them.  Putting the frames
      and the curves on ONE time axis is the whole idea of the dataset in a
      single picture: at every tick, this is what it saw and this is what it did.
      The gripper gets its own pane because it is not a pose, it is a decision.

    python3 03_plot_the_curve.py
"""

import warnings
warnings.filterwarnings('ignore')    # this box has two matplotlibs; the 3D one is not ours

import matplotlib
matplotlib.use('Agg')                # write a file; there is no window here

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
EPISODE = 1
STRIP = 12                           # wrist frames along the top

ds = LeRobotDataset(REPO_ID, root='dataset')
ep = ds.meta.episodes[EPISODE]
start, length = ep['dataset_from_index'], ep['length']
names = [n.replace('ur5e_', '').replace('_joint', '').replace('gripper_robotiq_85_left_knuckle', 'gripper')
         for n in ds.meta.info['features']['action']['names']]

action = np.array([ds[start + t]['action'].numpy() for t in range(length)])
time = np.arange(length) / ds.fps

fig, (top, arm, grip) = plt.subplots(
    3, 1, figsize=(18, 4.8), dpi=100, height_ratios=[1.15, 2.2, 1.3],
    gridspec_kw={'hspace': 0.34, 'left': 0.045, 'right': 0.995, 'top': 0.99, 'bottom': 0.09})

# --- top: the frames, stretched across the same time axis as the curves
steps = np.linspace(0, length - 1, STRIP).round().astype(int)
row = np.hstack([ds[start + s]['observation.images.wrist'].numpy().transpose(1, 2, 0)
                 for s in steps])
top.imshow(row, extent=[0, time[-1], 0, 1], aspect='auto')
top.set_yticks([]); top.set_xticks([])
top.set_ylabel('saw', rotation=0, ha='right', va='center', fontsize=11)

# --- middle: the six arm joints.  One line per joint, all in radians.
for i in range(6):
    arm.plot(time, action[:, i], lw=1.6, label=names[i])
arm.set_ylabel('did\n[rad]', rotation=0, ha='right', va='center', fontsize=11)
arm.legend(ncol=6, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
arm.grid(alpha=0.25); arm.set_xlim(0, time[-1]); arm.set_xticklabels([])

# --- bottom: the gripper alone - a square wave, not a trajectory
grip.plot(time, action[:, 6], lw=2.0, color='crimson', label='gripper command')
grip.fill_between(time, 0, action[:, 6], color='crimson', alpha=0.12)
grip.set_ylabel('grip', rotation=0, ha='right', va='center', fontsize=11)
grip.set_xlabel('time [s]'); grip.grid(alpha=0.25); grip.set_xlim(0, time[-1])
grip.legend(fontsize=9, loc='center left', frameon=False)

# The two instants that matter, drawn through all three panes: at these ticks
# the label stops being a trajectory and becomes a decision.
close = int(np.argmax(action[:, 6] > 0.25))
release = close + int(np.argmax(action[close:, 6] < 0.25))
for step, tag in ((close, 'CLOSE'), (release, 'OPEN')):
    for ax in (top, arm, grip):
        ax.axvline(step / ds.fps, color='crimson', lw=1.0, ls='--', alpha=0.55)
    grip.annotate(f'{tag} @ {step / ds.fps:.1f}s', (step / ds.fps, 0.25),
                  xytext=(6, 0), textcoords='offset points', fontsize=9, color='crimson')

fig.savefig('out/action_curves.png')
print(f'episode {EPISODE}: {length} frames x 7 action channels, {ds.fps} Hz  ->  '
      f'{length * 7} numbers over {time[-1]:.1f} s')
print(f'arm joints span {action[:, :6].min():+.2f} .. {action[:, :6].max():+.2f} rad;  '
      f'the gripper only ever takes '
      + ' or '.join(f'{v:.1f}' for v in sorted(set(np.round(action[:, 6], 3)))))
print(f'the only two decisions in 9.6 s: CLOSE at {close / ds.fps:.1f} s, '
      f'OPEN at {release / ds.fps:.1f} s.  Everything else is where to be.')
print('wrote out/action_curves.png   <- top: what it saw.  below: what it did.')
