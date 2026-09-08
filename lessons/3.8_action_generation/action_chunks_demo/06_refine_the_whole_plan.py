"""Lesson 3.8 - step 6: diffusion - start rough, refine the whole plan.

WHAT  Start from pure noise across all 50 actions at once and clean it up in a
      few passes, printing how far it still is from the real plan each time.

WHY   Slide 3's other style, and its sculpting analogy made concrete.  The
      autoregressive model writes the plan left to right; this one shapes the
      whole block at once and takes several passes to make it smooth.  That is
      why it sees the whole path - and why it costs more compute per plan.

      IMPORTANT: this is not a trained model.  A real one has to LEARN what to
      remove at each pass; here we simply move toward the answer, to show the
      shape of the procedure.  SmolVLA uses flow matching, of this family.

    python3 06_refine_the_whole_plan.py
"""

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from chunks import FPS, chunk, episodes

HORIZON, PASSES, RATE = 50, 8, 0.5       # 8 passes, halving the gap each time
SHOW = (0, 1, 2, 4, 8)
rng = np.random.default_rng(0)

target = chunk(episodes()[1], 0, HORIZON)
rough = rng.normal(target.mean(0), target.std(0) + 0.3, size=target.shape)

history = [rough.copy()]
plan = rough.copy()
for _ in range(PASSES):
    plan += RATE * (target - plan)       # a trained model LEARNS this step
    history.append(plan.copy())

print(f'diffusion does not write the plan left to right - it starts from noise across')
print(f'all {HORIZON} actions at once, and cleans the whole block up in a few passes:')
for k in SHOW:
    gap = np.degrees(np.sqrt(((history[k] - target) ** 2).mean()))
    label = 'pure noise' if k == 0 else ''
    print(f'   pass {k}   {label:<12s} {gap:6.2f} deg from the plan that was really flown')
print('a trained model learns what to clean off at each pass; this just shows the shape.')

fig, ax = plt.subplots(figsize=(16, 4.0), dpi=100)
time = np.arange(HORIZON) / FPS
for k in SHOW:
    ax.plot(time, history[k][:, 0], lw=1.4, alpha=0.75, label=f'pass {k}')
ax.plot(time, target[:, 0], lw=3.0, color='k', ls='--', label='the real plan')
ax.set_xlabel('time into the chunk [s]'); ax.set_ylabel('shoulder_pan [rad]')
ax.set_title('the whole 50-action chunk, refined pass by pass', fontsize=11)
ax.legend(ncol=6, fontsize=10, loc='upper right'); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig('out/refine.png')
print('wrote out/refine.png')
