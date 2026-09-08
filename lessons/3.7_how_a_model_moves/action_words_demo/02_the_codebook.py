"""Lesson 3.7 - step 2: build the action vocabulary, the way OpenVLA does.

WHAT  Cut every joint's range into 256 steps, and print how wide one step is.

WHY   To predict an action "like a word" you first need a vocabulary OF
      actions.  OpenVLA makes one by binning each joint into 256 values.  The
      question that decides whether the idea is any good is right here: how
      much movement does one bin hide?  If a bin were 5 degrees wide the robot
      would be useless.  Look at the last column.

    python3 02_the_codebook.py
"""

import numpy as np

from codebook import BINS, NAMES, all_actions, build

actions = all_actions()
low, high = build(actions)
step = (high - low) / (BINS - 1)          # one bin, in radians

print(f'{len(actions)} actions -> a vocabulary of {BINS} words per joint:')
for i, name in enumerate(NAMES):
    span = f'{low[i]:+.3f} .. {high[i]:+.3f} rad'
    fine = '' if i == 6 else f'  = {np.degrees(step[i]):.3f} deg'
    print(f'  {name:<14s} middle 98% is {span:<22s} one bin = {step[i]:.5f} rad{fine}')
