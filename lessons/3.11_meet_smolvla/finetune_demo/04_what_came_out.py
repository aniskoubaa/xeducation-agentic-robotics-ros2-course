"""Lesson 3.11 - what a finished fine-tune leaves on disk.

WHAT  List the checkpoints the 20,000-step run wrote, then diff the fine-tuned
      config against the base one it started from.

WHY   Training does not just change weights.  It rewrites the contract: the
      model now declares seven action numbers instead of six, it knows our
      camera by name, and it carries the mean and spread of our own frames.
      That is why demo 3 can load this folder and nothing else.

    python3 04_what_came_out.py
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from smol import BASE, banner

RUN = Path('runs/ripeness')
last = RUN / 'checkpoints/last/pretrained_model'
tuned = json.loads((last / 'config.json').read_text())
base = json.loads((Path(snapshot_download(BASE)) / 'config.json').read_text())

saved = sorted(p.name for p in (RUN / 'checkpoints').iterdir() if p.name != 'last')
weights = (last / 'model.safetensors').stat().st_size
stats = load_file(last / 'policy_preprocessor_step_5_normalizer_processor.safetensors')

banner('What the run left on disk')
print(f'  checkpoints   {"  ".join(saved)}   (+ "last" -> {saved[-1]})')
print(f'  the policy    checkpoints/last/pretrained_model/  - {weights / 1e6:.0f} MB, the only folder demo 3 needs')
print(f'  to resume     checkpoints/last/training_state/    - optimizer, scheduler, rng\n')

print(f'  {"":14s}{"smolvla_base":34s}after fine-tuning')
print(f'  {"action":14s}{str(base["output_features"]["action"]["shape"][0]) + " numbers":34s}{tuned["output_features"]["action"]["shape"][0]} numbers - the gripper is back')
print(f'  {"cameras":14s}{", ".join(k.split(".")[-1] for k in base["input_features"] if "images" in k):34s}fed from observation.images.wrist')
print(f'  {"statistics":14s}{"its own pretraining mix":34s}the mean and spread of our {int(stats["action.count"])} frames')
