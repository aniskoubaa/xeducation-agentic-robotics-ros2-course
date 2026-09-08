"""Lesson 3.8 - step 5: autoregressive - and why small errors pile up.

WHAT  Push the SAME small per-step error through the two schemes and watch what
      each does with it over 50 steps.

WHY   Slide 3's cost of the autoregressive style: "small errors can pile up".
      That is not a mood, it is arithmetic.  Predicting step k FROM step k-1
      means every mistake is inherited: the error takes a random walk and grows.
      Emitting all 50 from one look at the world gives every step the same
      fresh starting point, so the error stays where it started.

      There is no model here.  We are not predicting anything - we are showing
      how a fixed error of 0.20 deg travels through the two arrangements.

    python3 05_step_by_step.py
"""

import numpy as np

HORIZON, PER_STEP, TRIALS = 50, 0.20, 2000     # deg of error at every single step
rng = np.random.default_rng(0)

slips = rng.normal(0, PER_STEP, size=(TRIALS, HORIZON))
rms = lambda x: np.sqrt((x ** 2).mean(axis=0))

# autoregressive: step k is built on step k-1, so it inherits every earlier slip
carried = rms(np.cumsum(slips, axis=1))
# whole chunk: every step comes from the same one look, so nothing is inherited
fresh = rms(slips)

print(f'a {PER_STEP:.2f} deg slip at every step, pushed through a {HORIZON}-action plan:')
print(f'   {"step":>6s}   {"autoregressive":>16s}   {"whole chunk at once":>21s}')
for k in (1, 10, 25, 50):
    print(f'   {k:6d}   {carried[k - 1]:12.2f} deg   {fresh[k - 1]:17.2f} deg')
print(f'by step {HORIZON} the step-by-step plan has drifted '
      f'{carried[-1] / fresh[-1]:.1f}x further from where it meant to be.')
print('it leans on its own last answer; the chunk leans on the world, once.')
