"""Lesson 3.11 - load the policy we fine-tuned, and read its contract.

WHAT  Point LeRobot at the checkpoint folder demo 2 produced and ask it three
      questions: how many action numbers, which camera, how big a chunk.

WHY   Loading a policy for inference is three lines: the weights, and the two
      processors that were saved beside them.  Get those three lines right and
      you never think about normalisation again - the checkpoint remembers the
      mean and spread of our own 4228 frames.

    python3 01_load_the_policy.py
"""

import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import MODEL, banner, quiet

with quiet():                                                # it chatters while it loads
    policy = SmolVLAPolicy.from_pretrained(MODEL)            # the weights
    pre, post = make_pre_post_processors(policy.config, MODEL)   # in, and out
policy.eval()

rename = pre.steps[0].rename_map
banner(f'Loaded {MODEL}/ - {sum(p.numel() for p in policy.parameters()):,} parameters')
print(f'  device      {next(policy.parameters()).device}, {next(policy.parameters()).dtype}')
print(f'  takes       {list(rename)[0]}  ->  {list(rename.values())[0]}')
print(f'  returns     {policy.config.chunk_size} x {policy.config.output_features["action"].shape[0]} - the arm and the gripper, 5 s ahead')
print(f'  un-normalises with our own statistics, saved in the checkpoint:')
stats = post.steps[0].stats['action']
print(f'    action mean  {"  ".join(f"{v:6.3f}" for v in stats["mean"])}')
print(f'    action std   {"  ".join(f"{v:6.3f}" for v in stats["std"])}')
