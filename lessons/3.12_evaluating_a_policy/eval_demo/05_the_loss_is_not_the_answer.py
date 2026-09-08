"""Lesson 3.12 - the training loss kept falling.  Did the policy keep improving?

WHAT  Score four saved checkpoints - steps 5,000 / 10,000 / 15,000 / 20,000 -
      on the same held-out episodes, and print each one beside the training
      loss it had reached.  Then score the last checkpoint a second time, so
      the model's own run-to-run spread is in the table too.

WHY   This is the trap the lesson is named after.  A loss curve falls smoothly
      and never tells you when to stop; the held-out error does.  The two lines
      part company early, and the last quarter of training buys less than the
      noise of measuring it.

    python3 05_the_loss_is_not_the_answer.py     # ~45 s, writes out/loss_vs_error.png
"""

import re
from pathlib import Path

import matplotlib
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smol import ARM, HELD_OUT, HORIZON, REPO_ID, TASK, banner, quiet

matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = 'runs/split_20000.log'
LOG_FREQ = 100                           # --log_freq: one line per 100 steps, in order
CHECKPOINTS = sorted(int(folder.name)                       # every checkpoint --save_freq
                     for folder in Path('checkpoints').iterdir()  # left behind
                     if folder.name.isdigit())
PROBE, STRIDE = list(HELD_OUT)[:3], 10   # three held-out episodes, one frame per second

dataset = LeRobotDataset(REPO_ID, root='dataset')
loss = np.array([float(m) for m in re.findall(r'loss:([0-9.]+)', Path(LOG).read_text())])


def score(step, times=1):
    """Mean arm error, in degrees, of the checkpoint saved at this step."""
    folder = f'checkpoints/{step:06d}/pretrained_model'
    with quiet():
        policy = SmolVLAPolicy.from_pretrained(folder)
        pre, post = make_pre_post_processors(policy.config, folder)
    policy.eval()
    rounds = []
    for _ in range(times):
        gaps = []
        for index in PROBE:
            meta = dataset.meta.episodes[index]
            first, length = meta['dataset_from_index'], meta['length']
            for start in range(0, length, STRIDE):          # the same sampling as 03
                frame = dataset[first + start]
                plan = post(policy.predict_action_chunk(pre({
                    'observation.images.wrist': frame['observation.images.wrist'],
                    'observation.state': frame['observation.state'],
                    'task': TASK})))[0].float().numpy()
                ahead = min(HORIZON, length - start)        # the episode may end first
                human = np.stack([dataset[first + start + t]['action'].numpy()
                                  for t in range(ahead)])
                gaps.append(np.degrees(np.abs(plan[:ahead, ARM] - human[:, ARM])).mean())
        rounds.append(float(np.mean(gaps)))
        print('.', end='', flush=True)
    return rounds


banner('The loss fell for 20,000 steps.  Did the policy keep getting better?')
print('  scoring every saved checkpoint on the same held-out frames ', end='', flush=True)
error = [score(step)[0] for step in CHECKPOINTS[:-1]]
last, again = score(CHECKPOINTS[-1], times=2)      # same weights, fresh flow-matching noise
error.append(last)
print()
print(f'    {"checkpoint":22s}{"training loss":>15s}{"held-out error":>18s}')
for step, scored in zip(CHECKPOINTS, error):
    print(f'    step {step:>6,}{"":11s}{loss[step // LOG_FREQ - 1]:>13.3f}{scored:>15.2f} deg')
print(f'    {"the same weights again":22s}{loss[-1]:>13.3f}{again:>15.2f} deg   '
      '<- flow matching starts from noise')
print(f'  after step {CHECKPOINTS[0]:,} the loss fell {1 - loss[-1] / loss[CHECKPOINTS[0] // LOG_FREQ - 1]:.0%}, '
      f'the held-out error {1 - error[-1] / error[0]:.0%}')
print(f'  the last {CHECKPOINTS[-1] - CHECKPOINTS[-2]:,} steps moved the error by '
      f'{abs(error[-1] - error[-2]):.2f} deg; scoring the same weights twice moves it '
      f'{abs(again - error[-1]):.2f}')

figure, axes = plt.subplots(figsize=(16, 4.2), dpi=100,
                            gridspec_kw={'left': 0.05, 'right': 0.945, 'top': 0.89, 'bottom': 0.13})
axes.plot((np.arange(len(loss)) + 1) * LOG_FREQ, loss, color='C1', lw=1.6,
          label='training loss  (what you watch while it trains)')
axes.set(yscale='log', xlabel='training step', ylabel='training loss  (log)',
         title='Two curves that do not say the same thing')
axes.grid(alpha=0.25)

held = axes.twinx()
held.plot(CHECKPOINTS, error, 'o-', color='C0', lw=2.2, ms=7,
          label='held-out action error  (what you should watch)')
held.scatter([CHECKPOINTS[-1]], [again], marker='x', s=90, color='C0', zorder=5)
held.annotate('the same weights, scored twice', (CHECKPOINTS[-1], again), xytext=(-165, 30),
              textcoords='offset points', fontsize=9,
              arrowprops=dict(arrowstyle='->', lw=1))
held.text(CHECKPOINTS[0], max(error) * 1.22,
          f'after step {CHECKPOINTS[0]:,} the loss falls another '
          f'{1 - loss[-1] / loss[CHECKPOINTS[0] // LOG_FREQ - 1]:.0%}, '
          f'while the held-out error falls {1 - error[-1] / error[0]:.0%}',
          fontsize=10, style='italic')
held.set(ylabel='held-out error  (degrees)', ylim=(0, max(error) * 1.35))
figure.legend(loc='lower left', bbox_to_anchor=(0.075, 0.20), fontsize=10)
figure.savefig('out/loss_vs_error.png')
print('  wrote out/loss_vs_error.png')
