"""Lesson 3.7 - step 4: what does chopping the motion up actually cost?

WHAT  Encode every action in the dataset, decode it straight back, and measure
      how far the round trip moved each joint.

WHY   The obvious objection to action words is that a robot moves smoothly and
      integers do not.  So measure it.  The error is the width of half a bin -
      hundredths of a degree - which is far below what the arm can even repeat.
      That is why the discrete trick is allowed to work at all.

      Watch the last column too: the gripper only ever needs TWO of its 256
      words.  Open and closed.  A vocabulary can be far bigger than the thing
      you are describing needs.

    python3 04_what_binning_costs.py
"""

import numpy as np

from codebook import NAMES, all_actions, build, decode, encode

actions = all_actions()
low, high = build(actions)
words = encode(actions, low, high)
back = decode(words, low, high)
error = np.degrees(np.abs(back - actions))       # round-trip error, in degrees

print(f'{len(actions)} actions encoded to words and decoded straight back:')
for i, name in enumerate(NAMES):
    print(f'  {name:<14s} average error {error[:, i].mean():.3f} deg   '
          f'worst {error[:, i].max():.3f} deg   words actually used {len(set(words[:, i])):3d} of 256')
print(f'the six arm joints are off by {error[:, :6].mean():.3f} deg on average - '
      'the arm cannot even repeat itself that precisely.')
