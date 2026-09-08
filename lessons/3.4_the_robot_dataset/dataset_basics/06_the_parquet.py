"""Lesson 3.4 - "where are the images?"  Inside the table.

The slide shows the general v3.0 layout, where camera frames go to videos/ as
MP4.  Our reference set declares the camera as dtype "image" instead, so
LeRobot packs every frame into the parquet as PNG bytes - one file holds all
4228 frames, states and actions together.  That is why there is no videos/.

    python3 06_the_parquet.py
"""

import json
import os

import pyarrow.parquet as pq

DATASET = 'dataset'
DATA = os.path.join(DATASET, 'data', 'chunk-000', 'file-000.parquet')

f = pq.ParquetFile(DATA)
size = os.path.getsize(DATA) / 1e6
shape = json.load(open(os.path.join(DATASET, 'meta', 'info.json'))
                  )['features']['observation.images.wrist']['shape']

print(f'data/chunk-000/file-000.parquet   {f.metadata.num_rows} rows x '
      f'{f.metadata.num_columns} columns   {size:.1f} MB   '
      f'{f.metadata.num_row_groups} row groups (one per episode)')
for name in f.schema_arrow.names[:3]:
    t = str(f.schema_arrow.field(name).type).replace(' ', '')
    note = '  <- the PNG bytes live INSIDE the table' if 'binary' in t else ''
    print(f'  {name:26s} {t[:34]:36s}{note}'.rstrip())
print('  timestamp  frame_index  episode_index  index  task_index    the bookkeeping')

raw = f.metadata.num_rows * shape[0] * shape[1] * shape[2] / 1e6
print(f'raw pixels {f.metadata.num_rows} x {"x".join(map(str, shape))} = {raw:.0f} MB'
      f'  ->  {size:.1f} MB on disk  =  {raw / size:.0f}x smaller, and still one tidy file')
