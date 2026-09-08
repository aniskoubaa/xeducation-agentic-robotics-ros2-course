"""Lesson 3.4 - meta/ is the bookkeeping that makes data/ mean something.

data/ holds columns of numbers.  On its own that is not a dataset, it is a
spreadsheet.  meta/ is what turns it back into robot experience: the schema,
the instruction strings, where each episode starts and stops, and the
statistics training uses to rescale the numbers.

    python3 05_the_metadata.py
"""

import json
import os

import pyarrow.parquet as pq

DATASET = 'dataset'
meta = lambda *p: os.path.join(DATASET, 'meta', *p)

# ---- meta/info.json - the schema every tool reads first -------------------
info = json.load(open(meta('info.json')))
print(f'meta/info.json      LeRobot {info["codebase_version"]} | {info["robot_type"]} | '
      f'{info["fps"]} fps | {info["total_episodes"]} episodes | {info["total_frames"]} frames | '
      f'{len(info["features"])} features')

# ---- meta/tasks.parquet - the language, stored once and referenced by id ---
tasks = pq.read_table(meta('tasks.parquet')).to_pydict()
print('meta/tasks.parquet ', ' | '.join(f'task_index {i} -> {t!r}'
                                        for i, t in zip(tasks['task_index'], tasks['task'])),
      '  the L in V-L-A')

# ---- meta/episodes/ - where each demonstration starts and stops ------------
eps = pq.read_table(meta('episodes', 'chunk-000', 'file-000.parquet'),
                    columns=['episode_index', 'length', 'dataset_from_index',
                             'dataset_to_index']).to_pylist()
print('meta/episodes/     ', '   |   '.join(
    f'ep {e["episode_index"]}: {e["length"]} frames, rows {e["dataset_from_index"]}..'
    f'{e["dataset_to_index"] - 1}' for e in eps[:2]), '  ...49 rows in all')

# ---- meta/stats.json - the numbers training normalises with ----------------
st = json.load(open(meta('stats.json')))['action']
print(f'meta/stats.json     action  mean {[round(x, 2) for x in st["mean"][:4]]}..  '
      f'std {[round(x, 2) for x in st["std"][:4]]}..   one pair per number')
print('data/ alone is a table of numbers.  meta/ is what says what the numbers mean.')
