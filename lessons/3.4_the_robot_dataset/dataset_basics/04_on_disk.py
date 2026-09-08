"""Lesson 3.4 - the same dataset, seen as a folder on disk.

No library this time - just files.  A LeRobot v3.0 dataset is a standard,
shareable directory: meta/ is the bookkeeping, data/ is the frames.  Open the
files in the Explorer while this prints; there is nothing hidden.

    python3 04_on_disk.py
"""

import json
import os

DATASET = 'dataset'

def mb(*parts):
    p = os.path.join(DATASET, *parts)
    if os.path.isfile(p):
        return os.path.getsize(p) / 1e6
    return sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs) / 1e6

info = json.load(open(os.path.join(DATASET, 'meta', 'info.json')))

print(f'dataset/   LeRobot {info["codebase_version"]}   robot {info["robot_type"]}   '
      f'{info["fps"]} fps   {mb():.0f} MB total')
print(f'  meta/info.json           the schema: {len(info["features"])} features with their '
      f'shapes, and {info["total_episodes"]} episodes')
print('  meta/stats.json          min/max/mean/std per feature - training normalises with these')
print('  meta/tasks.parquet       the language: task_index -> the instruction string')
print(f'  meta/episodes/           {info["total_episodes"]} rows: each episode\'s length and its '
      'first/last row in data/')
print(f'  data/chunk-000/          the frames themselves: {info["total_frames"]} rows, '
      f'{mb("data"):.1f} MB')
