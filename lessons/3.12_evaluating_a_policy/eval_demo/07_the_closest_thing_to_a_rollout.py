"""Lesson 3.12 - replay every held-out episode the way a robot would run it.

WHAT  For each of the nine held-out episodes: look at the recorded frame, plan
      50 actions, keep the first 25, look again.  Then ask the one question a
      column of averages cannot answer - did the gripper close at the right
      moment, and only then?

WHY   The grasp is the task.  It happens in a tenth of a second and it is worth
      more than every other step put together, so it deserves to be counted,
      not averaged.  This is the strongest evidence offline data can give, and
      the last section explains why it still is not a success rate.

    python3 06_the_closest_thing_to_a_rollout.py
"""

import time

import matplotlib
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smol import ARM, HELD_OUT, HUMAN, MODEL, REPO_ID, TASK, banner, quiet

matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXECUTE = 25                             # keep half of every plan - lesson 3.8's rhythm
CLOSED = 0.25                            # the gripper command swings between 0.0 and 0.5

dataset = LeRobotDataset(REPO_ID, root='dataset')
with quiet():
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    pre, post = make_pre_post_processors(policy.config, MODEL)
policy.eval()


def around_the_grasp(flown, human, radius=5):
    """Arm error in the second the gripper shuts - the moment the task turns on."""
    shut = grip(human[:, 6])[0]
    window = slice(max(shut - radius, 0), shut + radius)
    return np.degrees(np.abs(flown[window, ARM] - human[window, ARM])).mean()


def grip(trace):
    """When the gripper shut, and when it let go again, in steps.  -1 if never."""
    shut = np.flatnonzero(trace > CLOSED)
    if not len(shut):
        return -1, -1, 0
    holds = np.split(shut, np.flatnonzero(np.diff(shut) > 1) + 1)   # separate closed intervals
    return holds[0][0], holds[0][-1] + 1, len(holds)


rollouts, calls, spent = {}, 0, 0.0
for index in HELD_OUT:
    meta = dataset.meta.episodes[index]
    first, length = meta['dataset_from_index'], meta['length']
    human = np.stack([dataset[first + t]['action'].numpy() for t in range(length)])
    flown = np.zeros_like(human)
    for start in range(0, length, EXECUTE):
        frame = dataset[first + start]
        clock = time.perf_counter()
        chunk = post(policy.predict_action_chunk(pre({
            'observation.images.wrist': frame['observation.images.wrist'],
            'observation.state': frame['observation.state'], 'task': TASK})))[0].float().numpy()
        spent += time.perf_counter() - clock
        calls += 1
        kept = min(EXECUTE, length - start)
        flown[start:start + kept] = chunk[:kept]
    rollouts[index] = (flown, human)

seconds = sum(len(h) for _, h in rollouts.values()) / dataset.fps
arm = np.array([np.degrees(np.abs(f[:, ARM] - h[:, ARM])).mean() for f, h in rollouts.values()])
closed = np.array([grip(f[:, 6])[0] >= 0 for f, _ in rollouts.values()])
timing = np.array([[a - b for a, b in zip(grip(f[:, 6])[:2], grip(h[:, 6])[:2])]
                   for f, h in rollouts.values()]) / dataset.fps
extra = sum(grip(f[:, 6])[2] > 1 for f, _ in rollouts.values())
critical = np.array([around_the_grasp(f, h) for f, h in rollouts.values()])

banner("The closest thing to a rollout we can run without a robot")
print(f"  {len(rollouts)} held-out episodes at lesson 3.8's rhythm - plan 50, execute {EXECUTE}, "
      f'look again')
print(f'    {calls} model calls, {spent:.1f} s of thinking, for {seconds:.1f} s of motion')
print()
row = lambda label, value: print(f'    {label:31s}{value}')
row('arm, over the whole episode', f'{arm.mean():.2f} deg per joint per step'
    f'     two humans differ by {HUMAN}')
row('in the second around the grasp', f'{critical.mean():.2f} deg per joint per step'
    '     the moment the whole task turns on')
row('the gripper closes', f'{closed.sum()} of {len(rollouts)} episodes'
    f'                worst offset {np.abs(timing[:, 0]).max() * 1000:.0f} ms')
row('and lets go again', f'{closed.sum()} of {len(rollouts)} episodes'
    f'                worst offset {np.abs(timing[:, 1]).max() * 1000:.0f} ms'
    f', {extra} spurious closes')
print()
print('  Nine for nine - on recordings.  That is not yet a success rate.')

figure, (traces, bars, offsets) = plt.subplots(
    1, 3, figsize=(16, 4.2), dpi=100, width_ratios=[2, 1.35, 0.9],
    gridspec_kw={'wspace': 0.24, 'left': 0.045, 'right': 0.99, 'top': 0.88, 'bottom': 0.16})

for order, (index, (flown, human)) in enumerate(rollouts.items()):
    clock = np.arange(len(human)) / dataset.fps
    traces.plot(clock, human[:, 6], color='0.55', lw=5, alpha=0.5)
    traces.plot(clock, flown[:, 6], color=f'C{order % 9}', lw=1.4, label=f'ep {index}')
traces.axhline(CLOSED, color='k', ls=':', lw=1)
traces.set(xlabel='seconds into the episode', ylabel='gripper command  (rad)', ylim=(-0.06, 0.62),
           title='thin: the policy.   thick grey: the human.   all 9 held-out episodes')
traces.legend(ncol=5, fontsize=8, loc='lower center')

width = 0.38
bars.bar(np.arange(len(arm)) - width / 2, arm, width, label='whole episode')
bars.bar(np.arange(len(arm)) + width / 2, critical, width, label='the second around the grasp')
bars.axhline(HUMAN, color='k', ls=':', lw=1.2)
bars.text(-0.55, HUMAN, 'two humans differ by this much', fontsize=8, va='center',
          bbox=dict(fc='white', ec='none', pad=1))
bars.set(xticks=range(len(arm)), xticklabels=[str(i) for i in HELD_OUT],
         ylim=(0, max(arm.max(), critical.max()) * 1.3),
         xlabel='held-out episode', ylabel='mean arm error  (degrees)',
         title='the whole episode, and the one second the grasp happens in')
bars.legend(fontsize=9, loc='upper right')

offsets.axhline(0, color='k', lw=1)
for column in (0, 1):
    offsets.scatter(np.full(closed.sum(), column) + np.linspace(-0.15, 0.15, closed.sum()),
                    timing[closed, column] * 1000, s=34)
offsets.set(xticks=[0, 1], xticklabels=['closing', 'letting go'], xlim=(-0.6, 1.6),
            ylim=(-260, 260), ylabel='policy minus human  (ms)', title='when the grasp happened')
offsets.text(0.5, 0.95, f'largest miss {np.abs(timing[closed]).max() * 1000:.0f} ms\n'
             'one robot step is 100 ms', fontsize=9, ha='center', va='top',
             transform=offsets.transAxes)
for panel in (traces, bars, offsets):
    panel.grid(alpha=0.25)
figure.savefig('out/gripper.png')
print('  wrote out/gripper.png')
