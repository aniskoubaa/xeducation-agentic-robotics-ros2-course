"""Lesson 3.11 - drive a whole episode the way a robot would.

WHAT  Walk through one full episode.  Every 25 steps, hand the policy the frame
      recorded at that instant, take the chunk of 50 it returns, and keep the
      first 25.  Four model calls cover the whole 9.7 s.  Plot the result
      against what the human commanded.

WHY   Lesson 3.8's rhythm - look, plan 50, fly 25, look again - with a real
      model doing the planning.  Note what this is NOT: the observations come
      from the recording, not from a robot that followed these actions.  That
      is the honest limit of an offline check, and 3.12 is about closing it.

    python3 04_a_whole_episode.py
"""

from smol import MODEL, REPO_ID, TASK, banner, quiet

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

EPISODE, EXECUTE = 1, 25             # keep 25 of every 50 planned - lesson 3.8's rhythm
JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']

ds = LeRobotDataset(REPO_ID, root='dataset')
episode = ds.meta.episodes[EPISODE]
first, length = episode['dataset_from_index'], episode['length']
truth = np.stack([ds[first + t]['action'].numpy() for t in range(length)])

with quiet():
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    pre, post = make_pre_post_processors(policy.config, MODEL)
policy.eval()

flown, calls, spent = np.zeros_like(truth), [], 0.0
for start in range(0, length, EXECUTE):
    frame = ds[first + start]                       # what the robot would be seeing
    clock = time.perf_counter()
    chunk = post(policy.predict_action_chunk(pre({
        'observation.images.wrist': frame['observation.images.wrist'],
        'observation.state': frame['observation.state'], 'task': TASK})))[0].float().numpy()
    spent += time.perf_counter() - clock
    kept = min(EXECUTE, length - start)
    flown[start:start + kept] = chunk[:kept]
    calls.append((start, kept))

seconds = np.arange(length) / ds.fps
fig, (arm, grip) = plt.subplots(2, 1, figsize=(16, 5.0), dpi=100, height_ratios=[2.2, 1],
                                sharex=True, gridspec_kw={'hspace': 0.12, 'left': 0.05,
                                                          'right': 0.995, 'top': 0.93, 'bottom': 0.10})
for j, name in enumerate(JOINTS):
    arm.plot(seconds, truth[:, j], color=f'C{j}', lw=3.0, alpha=0.30)
    arm.plot(seconds, flown[:, j], color=f'C{j}', lw=1.5, label=name)
grip.plot(seconds, truth[:, 6], color='C7', lw=3.0, alpha=0.35)
grip.plot(seconds, flown[:, 6], color='C7', lw=1.6)
for start, _ in calls:
    for ax in (arm, grip):
        ax.axvline(start / ds.fps, color='0.4', ls=':', lw=1.1)
arm.set_ylabel('arm joints (rad)'); grip.set_ylabel('gripper'); grip.set_xlabel('seconds')
arm.set_title(f'episode {EPISODE} re-planned every {EXECUTE} steps - thin: the policy, thick pale: the human',
              fontsize=12)
arm.legend(ncol=6, fontsize=9, loc='upper left', frameon=False)
for ax in (arm, grip):
    ax.grid(alpha=0.25)
fig.savefig('out/rollout.png')

banner(f'Episode {EPISODE}: {length} steps, {len(calls)} model calls')
for n, (start, kept) in enumerate(calls, 1):
    print(f'  call {n}   at t = {start / ds.fps:4.1f} s   planned 50, kept {kept}')
print(f'  {len(calls)} forward passes for {length / ds.fps:.1f} s of motion, in {spent:.1f} s of compute')
print(f'  mean gap {np.degrees(np.abs(flown - truth).mean()):.2f} deg per joint per step - written to out/rollout.png')
