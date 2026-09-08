"""Lesson 3.12 - before scoring anything, find out what a good score would be.

WHAT  Group the 49 demonstrations by the scene they show, then measure how far
      apart two demonstrations of the SAME scene are.

WHY   An action error means nothing on its own.  Half a degree is excellent if
      two humans doing the identical task differ by half a degree, and terrible
      if they differ by a hundredth.  This is the yardstick every number in the
      rest of this demo is read against - and it also tells us, honestly, how
      new our held-out episodes really are.

    python3 02_the_yardstick.py
"""

import numpy as np

from smol import ARM, HELD_OUT, banner, demonstrations

FPS = 10
GRIPPED = 0.1                            # the gripper command is a clean 0 -> 0.5 square

demos = demonstrations()
scene = {index: len(actions) for index, actions in demos.items()}    # length names the scene
names = dict(zip(sorted(set(scene.values())), 'AB'))

banner('How much do two demonstrations of the same scene differ?')
print('  the recording holds two scenes, and they alternate')
for length, name in names.items():
    members = [i for i in demos if scene[i] == length]
    one = demos[members[0]]
    grasp = int(np.argmax(one[:, 6] > GRIPPED))
    print(f'    scene {name}   {len(members)} episodes   {length} steps  {length / FPS:.1f} s'
          f'   pan {one[grasp, 0]:+.2f} rad   grasp at {grasp / FPS:.1f} s')
mix = '  +  '.join(f'{sum(1 for i in HELD_OUT if names[scene[i]] == n)} x {n}' for n in 'AB')
print(f'  held out    {mix}      both sides of the row - and no third scene')
print()
spread = [np.degrees(np.stack([demos[i] for i in demos if scene[i] == length])[:, :, ARM].std(0))
          for length in names]
print('  two demonstrations of the SAME scene differ by')
print(f'    typically  {np.mean([s.mean() for s in spread]):.2f} deg per joint per step'
      f'       worst single step  {max(s.max() for s in spread):.2f} deg')
print()
print('  That is the floor: a score near 0.4 deg means "as close as two humans are".')
