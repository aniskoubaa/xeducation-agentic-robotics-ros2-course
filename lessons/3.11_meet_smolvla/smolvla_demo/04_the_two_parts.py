"""Lesson 3.11 - what each half of SmolVLA is wired to do.

WHAT  Read the real config off the checkpoint and print the job description of
      the vision-language model and of the action expert.

WHY   Slide 2 says "a VLM that understands and an action expert that moves".
      Every one of these numbers is a setting in that checkpoint: how many
      cameras it will look at, how long an instruction it will read, how many
      of SmolVLM2's layers survived, how many actions come out at once.

    python3 04_the_two_parts.py
"""

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from smol import BASE

policy = SmolVLAPolicy.from_pretrained(BASE)
cfg = policy.config
cameras = [k for k in cfg.input_features if 'images' in k]

print(f'Part 1 - the vision-language model: understanding    ({cfg.vlm_model_name})')
print(f'  looks at   {len(cameras)} camera slots, each padded to {cfg.resize_imgs_with_padding[0]} x {cfg.resize_imgs_with_padding[1]}')
print(f'  reads      the instruction, up to {cfg.tokenizer_max_length} tokens')
print(f'  kept       {cfg.num_vlm_layers} of SmolVLM2\'s layers, and freezes the encoder: {cfg.freeze_vision_encoder}')

print('\nPart 2 - the action expert: movement')
print(f'  width      {cfg.expert_width_multiplier}x the language model, {len(policy.model.vlm_with_expert.lm_expert.layers)} layers of its own')
print(f'  outputs    {cfg.chunk_size} actions per call - {cfg.chunk_size / 10:.1f} s of motion at 10 Hz')
print(f'  method     flow matching: from noise to a plan in {cfg.num_steps} steps')
