"""Lesson 3.11 - the loss curve of the full 20,000-step fine-tune.

WHAT  Read the log that `finetune.sh` wrote during the real 20,000-step run and
      plot the training loss against the step number.

WHY   Watching three hundred steps go by tells you the run started.  It does
      not tell you it converged.  The curve does: a steep drop in the first
      thousand steps while the action expert learns the shape of this arm, then
      a long, slow settle.  Flat and low is what you are waiting for.

    python3 03_the_loss_curve.py
"""

from smol import banner                             # first - it quiets the loaders

import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LOG = 'runs/ripeness_20000.log'
LOG_FREQ = 100                     # --log_freq: one line per 100 steps, in order

# The log abbreviates big step numbers ("step:12K"), so count lines instead:
# line i is step (i+1) * LOG_FREQ.  The loss itself is printed in full.
loss = np.array([float(m) for m in re.findall(r'loss:([0-9.]+)', open(LOG).read())])
step = (np.arange(len(loss)) + 1) * LOG_FREQ

fig, ax = plt.subplots(figsize=(16, 4.4), dpi=100,
                       gridspec_kw={'left': 0.05, 'right': 0.99, 'top': 0.93, 'bottom': 0.13})
ax.plot(step, loss, color='C0', lw=1.6)
ax.axvline(1000, color='C3', ls='--', lw=1.2)
ax.annotate('warm-up ends: 1,000 steps', (1000, loss.max() * 0.8), xytext=(1400, loss.max() * 0.85),
            fontsize=10, color='C3')
ax.set_xlabel('training step'); ax.set_ylabel('flow-matching loss')
ax.set_title('SmolVLA fine-tuned on 49 episodes of "pick the red tomato"', fontsize=12)
ax.grid(alpha=0.3)
fig.savefig('out/loss_curve.png')

banner(f'{len(loss)} log lines from the real {step[-1]:,}-step run')
for at in (100, 1000, 5000, 10000, 20000):
    if at <= step[-1]:
        print(f'  step {at:>6,}   loss {loss[at // LOG_FREQ - 1]:.3f}')
print(f'\n  lowest {loss.min():.3f}, last {loss[-1]:.3f} - written to out/loss_curve.png')
