"""Lesson 3.11 - which of SmolVLA's 450M numbers fine-tuning is allowed to move.

WHAT  Load the base checkpoint and ask every parameter whether it requires a
      gradient.  Group the answers by sub-module.

WHY   "Fine-tuning a 450M model on 49 episodes" sounds impossible, and it would
      be if all 450M were free to move.  They are not: SmolVLA ships with
      train_expert_only = True, so the vision-language half is frozen solid and
      only the action expert - plus the small heads that face our robot - learn.
      That is why this fits in under 4 GB of GPU memory and finishes in an hour.

    python3 02_what_actually_trains.py
"""

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import BASE, banner

policy = SmolVLAPolicy.from_pretrained(BASE)
total = sum(p.numel() for p in policy.parameters())
moving = sum(p.numel() for p in policy.parameters() if p.requires_grad)


def report(label, module, note=''):
    free = sum(p.numel() for p in module.parameters() if p.requires_grad)
    all_ = sum(p.numel() for p in module.parameters())
    print(f'  {label:20s} {all_:>13,}   {"TRAINS" if free else "frozen"}   {note}'.rstrip())


inner = policy.model.vlm_with_expert
banner(f'Fine-tuning moves {moving:,} of {total:,} numbers  ({moving / total:.0%})')
report('vision_model', inner.vlm.model.vision_model, 'the image encoder is kept exactly as it is')
report('connector', inner.vlm.model.connector)
report('text_model', inner.vlm.model.text_model, 'the language half is kept whole')
report('action expert', inner.lm_expert, 'the half that produces movement')
print(f'  {"state / action heads":20s} {moving - sum(p.numel() for p in inner.lm_expert.parameters()):>13,}   TRAINS   re-fitted from 6 joints to our 7')
print(f'\n  train_expert_only = {policy.config.train_expert_only}. That is what makes 49 episodes enough.')
