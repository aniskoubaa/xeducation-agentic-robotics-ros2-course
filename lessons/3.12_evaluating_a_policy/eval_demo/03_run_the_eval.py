"""Lesson 3.12 - the offline check: run the policy over frames it never trained on.

WHAT  Walk three sets of episodes.  Twice a second, hand the policy the
      recorded camera image and joint state, and keep the 50 actions it plans
      beside the 50 the human actually commanded.

WHY   This is the cheap half of evaluation.  It answers "does it copy the
      expert?" - and only that.  We run it on held-out episodes, on training
      episodes, and on the base model that never saw our robot, because a
      single number tells you nothing without something to compare it to.

    python3 03_run_the_eval.py          # ~1 min on one GPU, writes out/eval.npz
"""

import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from smol import (BASE, EVAL, HELD_OUT, HORIZON, MODEL, REPO_ID, SEEN, STRIDE, TASK,
                  banner, quiet)

dataset = LeRobotDataset(REPO_ID, root='dataset')


def load(checkpoint):
    """A policy, its two processors, and whether it is one of ours or the stock one."""
    with quiet():
        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        pre, post = make_pre_post_processors(policy.config, checkpoint)
    return policy.eval(), pre, post, bool(pre.steps[0].rename_map)


def observe(frame, ours):
    """Our checkpoint learned our camera's name and our seventh joint; the base one
    knows neither, so it gets the image in all three of its slots and six joints."""
    image, state = frame['observation.images.wrist'], frame['observation.state']
    if ours:
        return {'observation.images.wrist': image, 'observation.state': state, 'task': TASK}
    return {f'observation.images.camera{slot}': image for slot in (1, 2, 3)} | {
        'observation.state': state[:6], 'task': TASK}


def sweep(label, model, episodes, stride):
    """Every `stride`-th frame of these episodes: what it planned, what the human did."""
    policy, pre, post, ours = model
    print(f'  {label:34s} ', end='', flush=True)
    plans, humans, when, clock = [], [], [], time.perf_counter()
    for index in episodes:
        meta = dataset.meta.episodes[index]
        first, length = meta['dataset_from_index'], meta['length']
        for start in range(0, length, stride):
            chunk = post(policy.predict_action_chunk(
                pre(observe(dataset[first + start], ours))))[0].float().numpy()
            plan = np.full((HORIZON, 7), np.nan, np.float32)
            plan[:, :chunk.shape[1]] = chunk[:HORIZON]          # the base model plans 6, not 7
            human = np.full((HORIZON, 7), np.nan, np.float32)   # the episode may end first
            for t in range(min(HORIZON, length - start)):
                human[t] = dataset[first + start + t]['action'].numpy()
            plans.append(plan)
            humans.append(human)
            when.append((index, start, length))
        print('.', end='', flush=True)
    spent = time.perf_counter() - clock
    print(f'  {len(plans):4d} predictions   {spent:5.1f} s')
    return np.stack(plans), np.stack(humans), np.array(when)


banner('Rolling the policy forward, one prediction every half second')
tuned, base = load(MODEL), load(BASE)
runs = {'held_out': sweep('fine-tuned   held out     ep 40-48', tuned, HELD_OUT, STRIDE),
        'seen': sweep('fine-tuned   trained on   ep  0-4', tuned, SEEN, STRIDE),
        'base': sweep('base model   held out     ep 40-48', base, HELD_OUT, STRIDE * 3)}
np.savez_compressed(EVAL, **{f'{k}_{n}': a for k, v in runs.items()
                             for n, a in zip(('plan', 'human', 'when'), v)})
print(f'\n  {EVAL}   {sum(len(v[0]) for v in runs.values())} chunks of 50 x 7, '
      'each beside the human answer for the same 5 seconds')
