"""Lesson 3.5 - what we just recorded, next to the dataset from lesson 3.4.

WHAT  Opens the episodes the auto-demonstrator just wrote AND the course's
      reference set, with the same LeRobot call, and prints them side by side.
      Then saves a filmstrip of the pick you watched being recorded.

WHY   The point of the whole lesson: teleoperation does not produce "some log
      file", it produces exactly the dataset we opened last lesson.  Same
      schema, same features, same instruction - only the episode count differs,
      and that is just a matter of repeating the demonstration.

    python3 check_dataset.py
"""
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

OUT = Path('out')                    # whatever the demonstrator just wrote
REF = Path('ref')                    # the reference set from lesson 3.4
STRIP = 6                            # frames in the filmstrip
TILE = 176                           # px per frame - 6 of them fit the editor pane


def summarise(root):
    """Open a LeRobot folder and describe it in one line."""
    # The folder is named <hf-user>__<dataset>; LeRobot wants that as a repo id.
    ds = LeRobotDataset(root.resolve().name.replace('__', '/'), root=str(root))
    return ds, (f'{ds.num_episodes:4d} episodes {ds.num_frames:6d} frames  '
                f'{ds.fps} fps  {ds.meta.info["robot_type"]}  '
                f'{ds.meta.tasks.index[0]!r}')


fresh_dir = sorted(p for p in OUT.iterdir() if p.is_dir())[-1]
fresh, fresh_line = summarise(fresh_dir)
ref, ref_line = summarise(REF)

print('just recorded ', fresh_dir)
print(' ', fresh_line)
print('the reference  ref/   (the dataset we opened in lesson 3.4)')
print(' ', ref_line)

feats = fresh.meta.info['features']
assert set(feats) == set(ref.meta.info['features']), 'schemas differ!'
img, state = feats['observation.images.wrist'], feats['observation.state']
print(f'same {len(feats)} features: images.wrist {"x".join(map(str, img["shape"]))} '
      f'(dtype {img["dtype"]}), state {state["dtype"]}[{state["shape"][0]}], action likewise')

# ---- the episode we just watched, as the model will see it ------------------
n = fresh.meta.episodes[0]['length']
tiles = [Image.fromarray((fresh[i]['observation.images.wrist']
                          .permute(1, 2, 0).numpy() * 255).astype('uint8'))
         .resize((TILE, TILE))
         for i in range(0, n, max(1, n // STRIP))][:STRIP]
strip = Image.new('RGB', (len(tiles) * (TILE + 6) - 6, TILE), 'white')
for i, tile in enumerate(tiles):
    strip.paste(tile, (i * (TILE + 6), 0))
strip.save('out/first_pick.png')
print(f'wrote out/first_pick.png   {len(tiles)} frames of episode 0 - '
      'the pick you just watched')
