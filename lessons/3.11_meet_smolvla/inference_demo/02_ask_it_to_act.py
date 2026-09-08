"""Lesson 3.11 - one observation in, one plan out, from the fine-tuned policy.

WHAT  The exact frame demo 1 gave to the untrained base model, now given to the
      model that has seen 49 episodes of this task.  Print the first actions of
      the plan next to what the human actually commanded at those instants.

WHY   This is inference, complete: build an observation dict, pass it through
      the preprocessor, call predict_action_chunk, pass the result through the
      postprocessor.  Four lines.  Everything else on screen is measurement.

    python3 02_ask_it_to_act.py
"""

import time

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import HORIZON, MODEL, REPO_ID, TASK, banner, one_frame, quiet, truth_after

ds = LeRobotDataset(REPO_ID, root='dataset')
frame = one_frame(ds)
truth = truth_after(ds)

with quiet():
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    pre, post = make_pre_post_processors(policy.config, MODEL)
policy.eval()

observation = {'observation.images.wrist': frame['observation.images.wrist'],
               'observation.state': frame['observation.state'],
               'task': TASK}

start = time.perf_counter()
plan = post(policy.predict_action_chunk(pre(observation)))[0].float().numpy()
elapsed = time.perf_counter() - start

banner(f'{HORIZON} actions in {elapsed:.2f} s, from one look at the world')
print(f'          {"shoulder_pan":>13s}{"lift":>9s}{"elbow":>9s}{"wrist_1":>9s}{"wrist_2":>9s}{"wrist_3":>9s}{"gripper":>9s}')
for step in (0, 10, 49):
    print(f'  t={step / 10:4.1f}s  ' + ''.join(f'{v:9.3f}' for v in plan[step]))
    print(f'  human    ' + ''.join(f'{v:9.3f}' for v in truth[step]))
print(f'\n  mean gap {np.degrees(np.abs(plan - truth).mean()):.1f} deg per joint per step - the base model was 40.')
