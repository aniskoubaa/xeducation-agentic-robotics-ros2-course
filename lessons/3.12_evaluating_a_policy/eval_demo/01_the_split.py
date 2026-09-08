"""Lesson 3.12 - what may we honestly score the policy on?

WHAT  Read the episode list out of two checkpoints: lesson 3.11's, and the one
      this lesson evaluates.  Every training run saves the list it was given.

WHY   An evaluation is only worth reading if the frames were withheld from
      training.  3.11 trained on all 49 episodes, which was convenient and left
      nothing to test on.  So we trained again - same steps, same batch, same
      seed - on 40 episodes, and kept 9 back.

    python3 01_the_split.py
"""

import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from smol import HELD_OUT, MODEL, MODEL_ALL, REPO_ID, TRAIN, banner

meta = LeRobotDatasetMetadata(REPO_ID, root='dataset')
frames = lambda episodes: sum(int(meta.episodes[e]['length']) for e in episodes)

banner('Which frames may we honestly score the policy on?')
print(f"  {REPO_ID.split('/')[1]:24s} {meta.total_episodes} episodes   "
      f'{meta.total_frames:,} frames   {meta.fps} Hz')
print()
for label, checkpoint in (("lesson 3.11's policy", MODEL_ALL), ('the policy we evaluate', MODEL)):
    config = json.loads((Path(checkpoint) / 'train_config.json').read_text())
    listed = config['dataset']['episodes']
    told = 'null  - every episode' if listed is None else f'[0..{listed[-1]}]  - {len(listed)} of them'
    print(f'  {label:24s} dataset.episodes = {told}')
print()
print(f'  trained on   episodes {TRAIN[0]:2d}-{TRAIN[-1]}   {frames(TRAIN):,} frames')
print(f'  held out     episodes {HELD_OUT[0]}-{HELD_OUT[-1]}   {frames(HELD_OUT)} frames'
      '    the policy has never been shown one of these')
print()
print(f"  Same {config['steps']:,} steps, batch {config['batch_size']}, seed {config['seed']} "
      'as 3.11 - only the episode list changed.')
