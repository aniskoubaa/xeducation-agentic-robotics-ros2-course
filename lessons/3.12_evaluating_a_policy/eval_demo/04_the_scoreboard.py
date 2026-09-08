"""Lesson 3.12 - the offline scoreboard, and why it needs three rows, not one.

WHAT  Turn out/eval.npz into numbers: how far each planned action is from the
      human's, for the held-out episodes, for episodes the policy trained on,
      and for the base model that never saw our robot.

WHY   "0.2 degrees" means nothing by itself.  Against the base model's 40 it
      looks like a triumph; against the 0.35 degrees that separates two human
      demonstrations it looks like the ceiling.  Both comparisons matter, and
      the gap between held-out and seen is the one that measures memorisation.

    python3 04_the_scoreboard.py
"""

import matplotlib
import numpy as np

from smol import ARM, EVAL, HUMAN, JOINTS, banner

matplotlib.use('Agg')
import matplotlib.pyplot as plt

data = np.load(EVAL)
SETS = {'held_out': 'fine-tuned   held out', 'seen': 'fine-tuned   trained on',
        'base': 'base model   held out'}


def error(name, steps=slice(None), joints=ARM):
    """How far the plan sits from the human, in degrees, with the padding dropped."""
    gap = data[f'{name}_plan'][:, steps, joints] - data[f'{name}_human'][:, steps, joints]
    return np.degrees(np.abs(gap))


banner('How far is the plan from the human?  (the six arm joints, in degrees)')
print(f'  {"":24s}{"first action":>14s}{"whole 5 s chunk":>18s}{"worst single step":>20s}')
for name, label in SETS.items():
    first, whole = error(name, 0), error(name)
    print(f'  {label:24s}{np.nanmean(first):>14.2f}{np.nanmean(whole):>18.2f}'
          f'{np.nanmax(whole):>20.1f}')
gap = np.nanmean(error('held_out')) - np.nanmean(error('seen'))
print(f'\n  held out minus trained on:  {gap:+.2f} deg'
      f'      two human demonstrations differ by {HUMAN}')

per_joint = {name: [np.nanmean(error(name, slice(None), j)) for j in range(7)] for name in SETS}
figure, axes = plt.subplots(figsize=(16, 4.2), dpi=100,
                            gridspec_kw={'left': 0.05, 'right': 0.995, 'top': 0.89, 'bottom': 0.11})
for offset, (name, label) in zip((-0.26, 0.0, 0.26), SETS.items()):
    bars = axes.bar(np.arange(7) + offset, per_joint[name], 0.25, label=label.replace('   ', ', '))
    if name == 'held_out':
        axes.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)
axes.text(6, 1.6, 'the base model\nhas no seventh joint', fontsize=9, ha='center', style='italic')
axes.axhline(HUMAN, color='k', ls=':', lw=1.2)
axes.text(1.35, HUMAN, 'two human demonstrations differ by this much', fontsize=9,
          va='center', bbox=dict(fc='white', ec='none', pad=1))
axes.set(yscale='log', xticks=range(7), ylim=(0.02, 400),
         ylabel='mean error   (degrees, log scale)',
         title='Where the error sits, joint by joint   -   9 held-out episodes, 164 predictions')
axes.set_xticklabels(JOINTS)
axes.legend(loc='upper left', fontsize=10)
axes.grid(axis='y', alpha=0.25)
figure.savefig('out/by_joint.png')
print('  wrote out/by_joint.png')
