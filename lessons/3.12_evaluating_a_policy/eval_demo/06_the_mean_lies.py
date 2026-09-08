"""Lesson 3.12 - the average hides exactly the moments you care about.

WHAT  Take the same held-out predictions and stop averaging them.  Look at the
      spread, at where in the episode the error appears, and at how it grows
      the further ahead the policy commits.

WHY   A mean of 0.3 degrees sounds like a solved task.  But a grasp is decided
      in about a tenth of a second, and an average over ten seconds cannot see
      it.  The percentiles and the two curves below are what tell you whether
      the policy is reliable or merely usually right.

    python3 05_the_mean_lies.py
"""

import matplotlib
import numpy as np

from smol import ARM, EVAL, HORIZON, STRIDE, banner, demonstrations

matplotlib.use('Agg')
import matplotlib.pyplot as plt

FPS = 10
CLOSED = 0.1

GRASP = {len(actions): int(np.argmax(actions[:, 6] > CLOSED))       # when the human closed the
         for actions in demonstrations().values()}                 # gripper, in each scene

data = np.load(EVAL)
plan, human, when = (data[f'held_out_{k}'] for k in ('plan', 'human', 'when'))
error = np.degrees(np.abs(plan[:, :, ARM] - human[:, :, ARM]))          # (N, 50, 6)
flat = error[~np.isnan(error)]

banner('The average is the least interesting number here')
print(f'  {flat.size:,} comparisons from {len(plan)} predictions on episodes it never saw')
for share, label in ((50, 'half of them are under '), (90, '9 in 10 are under      '),
                     (99, '99 in 100 are under    '), (100, 'and the worst one is   ')):
    print(f'    {label}{np.percentile(flat, share):7.2f} deg')
print(f'  the mean, {flat.mean():.2f} deg, sits above the median - that tail is doing the work')


def along_the_episode(length):
    """Mean error of the first planned action, moment by moment, in one scene."""
    scene = when[:, 2] == length
    return np.array([error[scene & (when[:, 1] == moment), 0, :].mean()
                     for moment in sorted(set(when[scene, 1]))])


profile = {length: along_the_episode(length) for length in GRASP}
span = max(len(curve) for curve in profile.values())
loudest = np.array([max(curve[moment] for curve in profile.values() if moment < len(curve))
                    for moment in range(span)])                 # the worst either scene managed
peak = int(loudest.argmax())
together = len({int(curve.argmax()) for curve in profile.values()}) == 1   # the same instant?
where = f'{peak * STRIDE / FPS:.1f} s in' + (', in both scenes,' if together else ',')
print(f'  nor is it spread evenly: {where} the mean jumps to '
      f'{loudest[peak]:.2f} deg, from '
      f'{np.median(np.concatenate(list(profile.values()))):.2f} elsewhere')

ahead = {name: np.nanmean(np.degrees(np.abs(data[f'{name}_plan'][:, :, ARM]
                                            - data[f'{name}_human'][:, :, ARM])), axis=(0, 2))
         for name in ('held_out', 'base')}
stale = {name: curve[-1] / curve[0] for name, curve in ahead.items()}     # how fast it decays
print(f"  and the plan goes stale gently: {ahead['held_out'][0]:.2f} deg at step 1, "
      f"{ahead['held_out'][-1]:.2f} at step 50 - {stale['held_out']:.1f}x, "
      f"against the base model's {stale['base']:.0f}x")

figure, (spread, timeline, horizon) = plt.subplots(
    1, 3, figsize=(16, 4.2), dpi=100,
    gridspec_kw={'wspace': 0.26, 'left': 0.05, 'right': 0.99, 'top': 0.88, 'bottom': 0.16})

spread.hist(flat, bins=np.logspace(-4, 1.2, 70), color='C0')
for share, style in ((50, '-'), (90, '--'), (99, ':')):
    spread.axvline(np.percentile(flat, share), color='k', ls=style, lw=1.2,
                   label=f'{share}th percentile  {np.percentile(flat, share):.2f} deg')
spread.axvline(flat.mean(), color='C3', lw=2, label=f'the mean  {flat.mean():.2f} deg')
spread.set(xscale='log', xlabel='error of one joint at one step  (degrees)',
           ylabel='how many', title='most steps are near-perfect; a few are not')
spread.legend(fontsize=8)

for length, grasp in GRASP.items():
    seconds = np.arange(len(profile[length])) * STRIDE / FPS
    timeline.plot(seconds, profile[length], marker='o', ms=3,
                  label=f'scene of {length / FPS:.1f} s')
    timeline.axvline(grasp / FPS, color=f'C{list(GRASP).index(length)}', ls=':', lw=1)
timeline.set_ylim(top=max(curve.max() for curve in profile.values()) * 1.12)
timeline.set(xlabel='seconds into the episode', ylabel='mean error  (degrees)',
             title='the error is not spread evenly through the episode')
timeline.text(0.02, 0.03, 'dotted: the human closed the gripper here', fontsize=8,
              transform=timeline.transAxes, ha='left', va='bottom')
timeline.legend(fontsize=9, loc='upper right')

for name, label in (('held_out', 'our fine-tuned policy'), ('base', 'the base model')):
    horizon.plot(np.arange(HORIZON) / FPS, ahead[name], lw=2, label=label)
horizon.axvline(25 / FPS, color='k', ls=':', lw=1.2)
horizon.text(2.6, 20, 'lesson 3.8 re-plans here', fontsize=9)
horizon.set(yscale='log', xlabel='seconds ahead, inside one 50-step plan',
            ylabel='mean error  (degrees, log)', title='how fast a plan goes stale')
horizon.legend(fontsize=9, loc='center right')
for panel in (spread, timeline, horizon):
    panel.grid(alpha=0.25)
figure.savefig('out/where_the_error_is.png')
print('  wrote out/where_the_error_is.png')
