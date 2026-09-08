"""Lesson 3.7 - step 6: the two ways a VLA can put the numbers out.

WHAT  Print the same instant twice: once as discrete action words, once as the
      continuous values it really was.

WHY   Slide 3's two bullets, side by side.  OpenVLA picks one of 256 choices
      per joint - a softmax over a vocabulary, exactly like choosing a word.
      SmolVLA and pi-zero hand the job to a small action expert that prints the
      numbers directly.  The difference is only the last layer; everything
      before it - see the image, read the instruction - is the same.

    python3 06_two_ways_out.py
"""

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from codebook import BINS, all_actions, build, decode, encode

EPISODE, STEP = 1, 54

low, high = build(all_actions())
ds = LeRobotDataset('raiseschool/raise_ripeness_sort_ref', root='dataset')
action = ds[ds.meta.episodes[EPISODE]['dataset_from_index'] + STEP]['action'].numpy()
words = encode(action, low, high)
rebuilt = decode(words, low, high)
gap = np.degrees(np.abs(rebuilt - action)).max()

row = lambda values, fmt: ''.join(f'{v:>{9}{fmt}}' for v in values)

print('one instant of one episode, two ways a VLA could produce it:')
print()
print(f'  DISCRETE    OpenVLA: one of {BINS} choices per joint - a softmax, like a word')
print(f'     {"words":<7s}' + row(words.tolist(), 'd'))
print(f'     {"means":<7s}' + row(rebuilt, '.3f') + f'   off by {gap:.3f} deg')
print()
print('  CONTINUOUS  SmolVLA, pi-zero: an action expert prints the numbers itself')
print(f'     {"values":<7s}' + row(action, '.3f') + '   exact')
print()
print('everything before the last layer is the same: see the image, read the instruction, act.')
