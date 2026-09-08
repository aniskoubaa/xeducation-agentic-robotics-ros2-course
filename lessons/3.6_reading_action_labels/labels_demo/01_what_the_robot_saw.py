"""Lesson 3.6 - what the robot saw, in time order.

WHAT  Pull twelve wrist-camera frames out of ONE episode of the reference set
      and lay them out left to right, with the timestamp under each one.

WHY   Slide 3 shows twelve real frames.  These are those frames - not an
      illustration, not a stock photo: the 224x224 images the policy reads at
      every step, in the order the robot saw them.  Read the strip and you can
      already tell the story: approach, descend, close, lift, carry, release.

    python3 01_what_the_robot_saw.py
"""

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image, ImageDraw

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
EPISODE = 1                          # the episode slide 4 plots: a full scan + pick
COLUMNS = 12                         # twelve frames, evenly spaced across the episode
TILE = 150                           # px per frame - twelve of them fit the editor pane

ds = LeRobotDataset(REPO_ID, root='dataset')
ep = ds.meta.episodes[EPISODE]
start, length = ep['dataset_from_index'], ep['length']
steps = np.linspace(0, length - 1, COLUMNS).round().astype(int)

# The camera is a feature like any other: ds[i] hands back the image as a
# CxHxW float tensor in 0..1, which is what the model gets - so put the
# channels last and scale back to bytes to look at it with human eyes.
sheet = Image.new('RGB', (COLUMNS * TILE, TILE + 22), 'white')
label = ImageDraw.Draw(sheet)
for col, step in enumerate(steps):
    chw = ds[start + step]['observation.images.wrist'].numpy()
    rgb = (chw.transpose(1, 2, 0) * 255).astype('uint8')
    sheet.paste(Image.fromarray(rgb).resize((TILE, TILE)), (col * TILE, 0))
    label.text((col * TILE + 6, TILE + 5), f't = {step / ds.fps:.1f} s', fill='black')

print(f'episode {EPISODE} of {ds.num_episodes}: {length} frames at {ds.fps} Hz'
      f'  =  {(length - 1) / ds.fps:.1f} s of one pick')
print(f'each frame  {rgb.shape[1]}x{rgb.shape[0]}x{rgb.shape[2]} uint8   '
      f'the model reads it as {chw.shape} float32 in 0..1')
print(f'showing {COLUMNS} of them, evenly spaced: t = '
      + ' '.join(f'{s / ds.fps:.1f}' for s in steps[:6]) + ' ... s')
out = Path('out/wrist_frames.png')
sheet.save(out)
print(f'wrote {out}   <- what the robot saw, left to right')
