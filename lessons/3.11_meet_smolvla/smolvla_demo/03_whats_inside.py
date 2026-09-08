"""Lesson 3.11 - count SmolVLA's parameters, part by part.

WHAT  Load the checkpoint and add up the parameters of each sub-module.

WHY   The slide says "~450M parameters: ~350M vision-language model plus
      ~100M action expert".  That is not a marketing round number - it is
      what the weights on disk actually contain, and you can check it in
      eight lines.  Understanding is the big half; moving is the small half.

    python3 03_whats_inside.py
"""

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import BASE, banner

policy = SmolVLAPolicy.from_pretrained(BASE)
total = sum(p.numel() for p in policy.parameters())
count = lambda module: sum(p.numel() for p in module.parameters())

vlm, expert = policy.model.vlm_with_expert.vlm, policy.model.vlm_with_expert.lm_expert
inner = vlm.model                       # vision tower, connector, language model
PARTS = {'vision_model': 'the image encoder',
         'connector': 'image patches -> language tokens',
         'text_model': 'the language model itself'}

banner(f'SmolVLA - {total:,} parameters')
print(f'  vision-language model {count(vlm):>13,} {count(vlm) / total:5.0%}   understands the image and the words')
for name, what in PARTS.items():
    print(f'    {name:19s} {count(getattr(inner, name)):>13,}         {what}')
print(f'    {"lm_head":19s} {count(vlm.lm_head):>13,}         frozen - SmolVLA never writes words')
print(f'  action expert         {count(expert):>13,} {count(expert) / total:5.0%}   turns understanding into joint angles')
rest = total - count(vlm) - count(expert)
print(f'  the robot-shaped edges {rest:>12,} {rest / total:5.0%}   joint angles in, joint angles out')
