"""Lesson 3.12 - the honest limits of everything measured so far.

WHAT  Ask how new the held-out episodes really are, by finding each one's
      nearest neighbour among the forty the policy trained on.

WHY   Every number in this demo is an offline number: it says the policy
      reproduces demonstrations it was not shown.  It cannot say the robot
      picks the tomato, and on this dataset it cannot even say the policy
      generalises - because the held-out episodes are repeats of two scenes it
      already knows.  Saying so is part of the evaluation.

    python3 07_what_it_does_not_prove.py
"""

import numpy as np

from smol import ARM, EVAL, HELD_OUT, TRAIN, banner, demonstrations

demos = demonstrations()
nearest = [min(np.degrees(np.abs(demos[held][:, ARM] - demos[trained][:, ARM])).mean()
               for trained in TRAIN if len(demos[trained]) == len(demos[held]))
           for held in HELD_OUT]
scored = np.load(EVAL)
policy = np.nanmean(np.degrees(np.abs(scored['held_out_plan'][:, :, ARM]
                                      - scored['held_out_human'][:, :, ARM])))

banner('What these numbers do not prove')
print('  every held-out episode has a near-twin among the forty the policy trained on')
print(f'    the nearest training episode is  {np.mean(nearest):.2f} deg away'
      f'      (least alike of the nine: {max(nearest):.2f})')
verdict = 'so a lookup table would have won' if np.mean(nearest) < policy else 'only just ahead of it'
print(f'    the policy itself scored         {policy:.2f} deg      {verdict}')
print('  That does not make the policy bad; it makes this held-out set too easy to be')
print('  evidence.  It holds repeats of two scenes, not a tomato in a place it never saw.')
print()
print('  The half that needs the robot: N trials at new positions and new lighting  ->  a')
print('  success rate;  the wrong, green grabs counted apart  ->  a safety number;  and')
print('  watch the rollouts, because the number says when it failed - the video says why.')
