"""Lesson 3.11 - run SmolVLA once, on a real frame from our dataset.

WHAT  Take one wrist-camera frame and the robot's joint angles at that moment,
      hand them to the base model together with the instruction, and print the
      plan that comes back.

WHY   This is the entire interface of a VLA, in six lines of code: an image, a
      state, a sentence in, a chunk of future actions out.  Notice the shape
      of what comes back - and notice that it has six numbers, not our seven.

    python3 05_one_look_one_plan.py
"""

import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import BASE, REPO_ID, TASK, banner, one_frame

frame = one_frame(LeRobotDataset(REPO_ID, root='dataset'))
policy = SmolVLAPolicy.from_pretrained(BASE)
pre, post = make_pre_post_processors(policy.config, BASE)

image = frame['observation.images.wrist']
observation = {'observation.images.camera1': image,      # our one camera, in the first slot
               'observation.images.camera2': image,
               'observation.images.camera3': image,
               'observation.state': frame['observation.state'][:6],
               'task': TASK}

start = time.perf_counter()
plan = post(policy.predict_action_chunk(pre(observation)))
elapsed = time.perf_counter() - start

banner('One look at the world, one plan out')
print(f'  in    a {image.shape[1]}x{image.shape[2]} wrist frame + {len(observation["observation.state"])} joint angles + "{TASK}"')
print(f'  out   {tuple(plan.shape)} - {plan.shape[1]} actions from one call, in {elapsed:.2f} s')
for step in range(3):
    joints = '  '.join(f'{v:7.3f}' for v in plan[0, step])
    print(f'    t = {step / 10:.1f} s   {joints}')
print('\n  Six numbers, not our seven: these weights have never met our arm. Fine-tuning is next.')
