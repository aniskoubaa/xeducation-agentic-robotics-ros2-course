"""Lesson 3.11 - demo 1's figure, redrawn with the fine-tuned policy.

WHAT  Identical frame, identical instruction, identical plot as
      `06_what_a_plan_looks_like.py` in demo 1 - only the weights have changed.

WHY   Fine-tuning is easiest to believe as a picture.  Before: a plausible arm
      motion that has nothing to do with this task.  After: the model's solid
      lines sit on top of the human's dashed ones, gripper included.  Same
      450M parameters; 20,000 steps on 49 episodes is the whole difference.

    python3 03_the_same_picture.py
"""

from smol import HORIZON, MODEL, REPO_ID, TASK, banner, one_frame, quiet, truth_after

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3', 'gripper']

ds = LeRobotDataset(REPO_ID, root='dataset')
frame = one_frame(ds)
truth = truth_after(ds)

with quiet():
    policy = SmolVLAPolicy.from_pretrained(MODEL)
    pre, post = make_pre_post_processors(policy.config, MODEL)
policy.eval()

image = frame['observation.images.wrist']
plan = post(policy.predict_action_chunk(pre({
    'observation.images.wrist': image,
    'observation.state': frame['observation.state'], 'task': TASK})))[0].float().numpy()

fig, (eye, curves) = plt.subplots(1, 2, figsize=(16, 4.2), dpi=100, width_ratios=[1, 3.6],
                                  gridspec_kw={'wspace': 0.12, 'left': 0.005, 'right': 0.995,
                                               'top': 0.90, 'bottom': 0.24})
eye.imshow(image.numpy().transpose(1, 2, 0)); eye.axis('off')
eye.set_title(f'what it looked at   "{TASK}"', fontsize=11)

seconds = np.arange(HORIZON) / ds.fps
for j, name in enumerate(JOINTS):
    curves.plot(seconds, truth[:, j], color=f'C{j}', lw=3.0, ls='-', alpha=0.30)
    curves.plot(seconds, plan[:, j], color=f'C{j}', lw=1.6, label=name)
curves.set_xlabel('seconds into the plan'); curves.set_ylabel('radians')
curves.set_title('thin: the fine-tuned model\'s 50 actions      thick pale: what the human did', fontsize=11)
curves.legend(ncol=7, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.19), frameon=False)
curves.grid(alpha=0.25)
fig.savefig('out/tuned_plan.png')

gap = np.degrees(np.abs(plan - truth).mean(axis=0))
banner('The same picture as demo 1, after 20,000 steps')
for j, name in enumerate(JOINTS):
    print(f'  {name:16s} off by {gap[j]:5.2f} deg on average' + ('   <- it learned to close the gripper' if name == 'gripper' else ''))
print(f'  written out/tuned_plan.png')
