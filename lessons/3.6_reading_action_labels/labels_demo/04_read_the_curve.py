"""Lesson 3.6 - read the curve back as a sentence.

WHAT  Walk the action channels frame by frame, group the frames where the same
      joints are moving, and print each group as one line of plain English.

WHY   Slide 4 says: read the curves as a story of motion.  This does exactly
      that, mechanically, with no hand-written script of the episode - the story
      comes out of the numbers.  Note what disambiguates it: the SAME shoulder
      sweep means "search" with the gripper open and "carry" with it closed.
      That is why a policy has to see the whole state, not one joint.

    python3 04_read_the_curve.py
"""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
EPISODE = 1
MOVING = 0.01                        # rad per frame: below this a joint is holding
SHORTEST = 3                         # frames; shorter runs are settling, not motion
PAN, LIFT, GRIP = 0, 1, 6

ds = LeRobotDataset(REPO_ID, root='dataset')
ep = ds.meta.episodes[EPISODE]
start, length = ep['dataset_from_index'], ep['length']
a = np.array([ds[start + t]['action'].numpy() for t in range(length)])
short = [n.replace('ur5e_', '').replace('_joint', '') for n in
         ds.meta.info['features']['action']['names']]

# Which joints are moving at each step -> the label of that step.
step_moves = [tuple(np.flatnonzero(np.abs(a[t + 1, :6] - a[t, :6]) > MOVING))
              for t in range(length - 1)]

runs, run_start = [], 0              # group consecutive steps with the same label
for t in range(1, len(step_moves) + 1):
    if t == len(step_moves) or step_moves[t] != step_moves[run_start]:
        runs.append((run_start, t, step_moves[run_start]))
        run_start = t


def describe(first, last, joints):
    """One phase in words. The gripper decides what a shoulder sweep MEANS."""
    holding = a[first, GRIP] > 0.25
    if len(joints) >= 3:
        return 'the whole arm', 'unfold from home toward the plant row'
    if joints == (PAN,):
        return short[PAN], ('carry it across' if holding else
                            'sweep the base across the row - the scan')
    if joints == (LIFT,):
        down = a[last, LIFT] > a[first, LIFT]      # toward 0 = the tool goes down
        return short[LIFT], ('reach down onto the tomato' if down else
                             'lift it clear' if holding else 'draw back up')
    return ' '.join(short[j] for j in joints), 'adjust'


# Two kinds of entry on one timeline: phases of motion, and gripper flips.
timeline = []
for first, last, joints in runs:
    # A run of wrists alone is the arm settling after a move, not a phase of it.
    if last - first < SHORTEST or not set(joints) & {PAN, LIFT, 2}:
        continue
    joint, verb = describe(first, last, joints)
    lead = joints[0]
    timeline.append((first, f' {first / ds.fps:4.1f} -{last / ds.fps:4.1f} s   {joint:<14s}'
                            f' {a[first, lead]:+.2f} ->{a[last, lead]:+.2f} rad   {verb}'))
for t in range(1, length):
    if abs(a[t, GRIP] - a[t - 1, GRIP]) > 0.25:
        verb = 'CLOSE - this is the grasp' if a[t, GRIP] > 0.25 else 'OPEN  - let it go'
        timeline.append((t, f' {t / ds.fps:4.1f} s         gripper        '
                            f'{a[t - 1, GRIP]:.1f} -> {a[t, GRIP]:.1f}         {verb}'))

print(f'episode {EPISODE}, {length} frames - read straight off the action channels:')
for _, line in sorted(timeline):
    print(line)
