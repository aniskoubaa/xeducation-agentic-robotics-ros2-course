"""Lesson 3.11 - draw the base model's plan against what the human really did.

WHAT  Same single observation as script 05.  Plot the 50 predicted joint angles
      beside the wrist frame the model looked at, and overlay - dashed - the
      50 actions the human teleoperator actually commanded from that moment.

WHY   This is the honest picture of an un-fine-tuned VLA.  SmolVLA has been
      pretrained on a lot of robots, but never on OUR arm and never on OUR
      task, so its plan is a plausible-looking arm motion that is simply not
      this one.  Keep this figure in mind: demo 3 redraws it after training.

    python3 06_what_a_plan_looks_like.py
"""

from smol import BASE, EPISODE, HORIZON, REPO_ID, START, TASK, banner, one_frame  # first - it quiets the loaders

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']

ds = LeRobotDataset(REPO_ID, root='dataset')
frame = one_frame(ds)
first = ds.meta.episodes[EPISODE]['dataset_from_index'] + START
truth = np.stack([ds[first + t]['action'].numpy() for t in range(HORIZON)])

policy = SmolVLAPolicy.from_pretrained(BASE)
pre, post = make_pre_post_processors(policy.config, BASE)
image = frame['observation.images.wrist']
plan = post(policy.predict_action_chunk(pre({
    'observation.images.camera1': image, 'observation.images.camera2': image,
    'observation.images.camera3': image,
    'observation.state': frame['observation.state'][:6], 'task': TASK})))
plan = plan[0].float().numpy()

fig, (eye, curves) = plt.subplots(1, 2, figsize=(16, 4.2), dpi=100, width_ratios=[1, 3.6],
                                  gridspec_kw={'wspace': 0.12, 'left': 0.005, 'right': 0.995,
                                               'top': 0.90, 'bottom': 0.24})
eye.imshow(image.numpy().transpose(1, 2, 0)); eye.axis('off')
eye.set_title(f'what it looked at   "{TASK}"', fontsize=11)

time = np.arange(HORIZON) / ds.fps
for j, name in enumerate(JOINTS):
    colour = f'C{j}'
    curves.plot(time, truth[:, j], color=colour, lw=1.1, ls='--', alpha=0.55)
    curves.plot(time, plan[:, j], color=colour, lw=2.0, label=name)
curves.set_xlabel('seconds into the plan'); curves.set_ylabel('radians')
curves.set_title('solid: the base model\'s 50 actions      dashed: what the human actually did', fontsize=11)
curves.legend(ncol=6, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.19), frameon=False)
curves.grid(alpha=0.25)
fig.savefig('out/base_plan.png')

gap = np.abs(plan - truth[:, :6]).mean()
banner('The base model has never met this arm')
print(f'  predicted {plan.shape[0]} actions x {plan.shape[1]} joints, against the {HORIZON} the human commanded')
print(f'  mean gap  {gap:.3f} rad  ({np.degrees(gap):.1f} deg) per joint per step')
print(f'  written   out/base_plan.png')
