"""Lesson 3.11 - the dataset, as the trainer will see it.

WHAT  Read our recording's own metadata and put it side by side with what
      `lerobot/smolvla_base` declares it expects.

WHY   Fine-tuning fails at the first step for exactly one boring reason: the
      names and shapes do not line up.  The base model wants three cameras
      called camera1..3 and a six-joint arm; we have one camera called `wrist`
      and seven numbers per action.  Seeing the mismatch first makes the two
      odd-looking flags in finetune.sh obvious instead of magic.

    python3 01_what_the_trainer_sees.py
"""

import json
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from smol import BASE, REPO_ID, banner

ds = LeRobotDataset(REPO_ID, root='dataset')
info = ds.meta.info
ours = [k for k in info['features'] if k.startswith('observation.images')]
base = json.loads((Path(snapshot_download(BASE)) / 'config.json').read_text())
theirs = [k for k in base['input_features'] if 'images' in k]

banner(f'Our recording: {info["total_episodes"]} episodes, {info["total_frames"]} frames at {info["fps"]} Hz')
print(f'  task      {ds.meta.tasks.index[0]!r}   (the only one in the set)')
print(f'  camera    {ours[0]:34s} {tuple(info["features"][ours[0]]["shape"])}')
print(f'  state     {"observation.state":34s} {info["features"]["observation.state"]["shape"][0]} joints')
print(f'  action    {"action":34s} {info["features"]["action"]["shape"][0]} joints\n')
print(f'  smolvla_base expects {len(theirs)} cameras named {", ".join(t.split(".")[-1] for t in theirs)} - we have one, named {ours[0].split(".")[-1]}')
print(f'  ... so training needs --rename_map, and it re-fits the heads from 6 joints to 7.')
