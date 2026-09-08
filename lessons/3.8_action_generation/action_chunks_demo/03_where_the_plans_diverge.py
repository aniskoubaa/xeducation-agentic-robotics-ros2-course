"""Lesson 3.8 - step 3: how long a chunk is TOO long - measured.

WHAT  Line up all 49 episodes.  25 of them fetch the tomato on the right, 24 on
      the left.  Print how far apart the two groups' plans are, second by
      second, and draw all 49 shoulder-pan traces on one axis.

WHY   Slide 2's trade-off: "if the chunk is too long, the robot commits to too
      much".  In this dataset the commitment is visible.  Every episode starts
      with the SAME unfold from home - nothing is decided yet - and only at
      about 2 seconds does the arm turn toward one tomato or the other.  A plan
      longer than that is promising which fruit to pick before looking.

    python3 03_where_the_plans_diverge.py
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from chunks import FPS, episodes, side

eps = episodes()
right = [e for e in eps if side(e) == 'right']
left = [e for e in eps if side(e) == 'left']
shared = min(min(len(e) for e in right), min(len(e) for e in left))

# The average plan of each group, compared frame by frame.
mean_right = np.mean([e[:shared] for e in right], axis=0)
mean_left = np.mean([e[:shared] for e in left], axis=0)
apart = np.degrees(np.abs(mean_right - mean_left).max(axis=1))
decides = int(np.argmax(apart > 2.0))     # first frame more than 2 deg apart

print(f'{len(eps)} episodes: {len(right)} fetch the tomato on the right, {len(left)} on the left')
for t in (10, 20, decides, 30, 40):
    tag = '  <- the plans split here' if t == decides else ''
    print(f'   t = {t / FPS:3.1f} s   the two groups are {apart[t]:5.2f} deg apart{tag}')
print(f'up to {decides / FPS:.1f} s every episode does the same thing: unfold from home.')
print(f'a chunk of 50 promises 5.0 s of motion - {(50 - decides) / FPS:.1f} s of it past the decision.')

fig, ax = plt.subplots(figsize=(16, 4.2), dpi=100)
for e in right:
    ax.plot(np.arange(len(e)) / FPS, e[:, 0], color='tab:blue', lw=1.0, alpha=0.5)
for e in left:
    ax.plot(np.arange(len(e)) / FPS, e[:, 0], color='tab:red', lw=1.0, alpha=0.5)
ax.axvspan(0, decides / FPS, color='0.85', alpha=0.6)
ax.text(decides / 2 / FPS, 0.75, 'identical so far\nnothing decided', ha='center', fontsize=11)
ax.axvline(decides / FPS, color='k', ls='--', lw=1.2)
ax.annotate(f'the decision, at {decides / FPS:.1f} s', (decides / FPS, -0.30),
            xytext=(10, 0), textcoords='offset points', fontsize=11)
ax.plot([], [], color='tab:blue', label=f'{len(right)} episodes: tomato on the right')
ax.plot([], [], color='tab:red', label=f'{len(left)} episodes: tomato on the left')
ax.set_xlabel('time [s]'); ax.set_ylabel('shoulder_pan [rad]')
ax.legend(loc='upper right', fontsize=10); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig('out/divergence.png')
print('wrote out/divergence.png')
