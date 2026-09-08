"""Lesson 3.11 - is one call fast enough to keep a 10 Hz robot moving?

WHAT  Time ten warm calls to `predict_action_chunk` and turn the answer into
      the only number that matters: how much of the robot's clock the policy
      spends thinking.

WHY   A VLA that is 100 ms too slow is a VLA that cannot run.  Chunking is what
      makes 450M parameters affordable here: one call buys 25 executed steps -
      2.5 seconds of motion at 10 Hz - so the model only has to think once
      every 2.5 s, not ten times a second.

    python3 05_fast_enough.py
"""

import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import MODEL, REPO_ID, TASK, banner, one_frame, quiet

CALLS, EXECUTE, FPS = 10, 25, 10

ds = LeRobotDataset(REPO_ID, root='dataset')
frame = one_frame(ds)
with quiet():
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    pre, post = make_pre_post_processors(policy.config, MODEL)
policy.eval()

observation = {'observation.images.wrist': frame['observation.images.wrist'],
               'observation.state': frame['observation.state'], 'task': TASK}

times = []
for call in range(CALLS + 1):                     # the first call warms the GPU up
    clock = time.perf_counter()
    post(policy.predict_action_chunk(pre(observation)))
    times.append((time.perf_counter() - clock) * 1000)
warm = np.array(times[1:])
budget = EXECUTE / FPS * 1000                     # ms of robot motion bought per call

banner(f'{CALLS} warm calls on this GPU')
print(f'  per call     {warm.mean():.0f} ms   (fastest {warm.min():.0f}, slowest {warm.max():.0f}; first, cold: {times[0]:.0f})')
print(f'  inside it    {policy.config.num_steps} flow-matching steps over all {policy.config.chunk_size} actions')
print(f'  it buys      {EXECUTE} executed steps = {budget / 1000:.1f} s of motion at {FPS} Hz')
print(f'  so the policy thinks for {warm.mean() / budget:.0%} of the robot\'s clock, and moves for the rest.')
print(f'\n  one action at a time would need a call every {1000 / FPS:.0f} ms; one call costs {warm.mean():.0f} ms.')
print(f'  chunking is what turns that race into thinking for {warm.mean() / budget:.0%} of the robot\'s clock.')
