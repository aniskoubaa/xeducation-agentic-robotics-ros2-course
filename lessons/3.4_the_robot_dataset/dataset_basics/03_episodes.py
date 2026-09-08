"""Lesson 3.3 - add time: frame -> episode -> dataset.

One frame is a single instant.  A robot does not act in an instant, so we grab
frames ten times a second and keep them in order.  One demonstration, start to
finish, is an EPISODE.  Many episodes are the DATASET.  That is the whole idea.

This also writes a filmstrip of episode 0, so you can see the sequence the
model is asked to imitate.

    python3 03_episodes.py
"""

from collections import Counter

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
DATASET = 'dataset'
EP = 0
STRIP = [0, 15, 30, 45, 60, 75]      # every 1.5 s of episode 0 - the six moments

ds = LeRobotDataset(REPO_ID, root=DATASET)
e = ds.meta.episodes[EP]             # one row of meta/episodes: where it starts, how long
start, n = e['dataset_from_index'], e['length']

print('frame    one instant  = image + state + action + task')
print(f'episode  {n} frames of episode {EP}, t = 0.0 .. {(n - 1) / ds.fps:.1f} s '
      f'at {ds.fps} Hz  = one whole pick')
print(f'dataset  {ds.num_episodes} episodes, {ds.num_frames} frames - every episode laid end to end')

lengths = Counter(ds.meta.episodes['length'])
print('lengths ', '   '.join(f'{k} frames x {v} episodes' for k, v in sorted(lengths.items())),
      '  (the longer ones scan twice)')

# ---- the episode as a picture: six frames, in time order ------------------
tiles = [Image.fromarray((ds[start + s]['observation.images.wrist']
                          .permute(1, 2, 0).numpy() * 255).astype('uint8')) for s in STRIP]
w, h = tiles[0].size
strip = Image.new('RGB', (len(tiles) * (w + 6) - 6, h), 'white')
for i, t in enumerate(tiles):
    strip.paste(t, (i * (w + 6), 0))
strip.save(f'out/episode_{EP:02d}_strip.png')
print(f'wrote out/episode_{EP:02d}_strip.png   t =',
      ' '.join(f'{s / ds.fps:.1f}' for s in STRIP), 's  - left to right, the pick')
