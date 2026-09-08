"""The episodes, and the chunks you can cut out of them.

A "chunk" sounds like a new kind of object.  It is not: it is H consecutive
actions - a slice of the same table lessons 3.4 to 3.7 have been reading.  The
model's job is to produce that slice from one look at the world.
"""

import numpy as np
import pyarrow.parquet as pq

TABLE = 'dataset/data/chunk-000/file-000.parquet'
FPS = 10
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2',
         'wrist_3', 'gripper']


def episodes():
    """Every episode as its own (frames x 7) array, in recording order."""
    table = pq.read_table(TABLE, columns=['action', 'episode_index'])
    actions = np.stack(table['action'].to_numpy(zero_copy_only=False))
    index = table['episode_index'].to_numpy()
    return [actions[index == i] for i in sorted(set(index))]


def chunk(actions, start, horizon):
    """The plan a model would emit at `start`: the next `horizon` actions.

    Short at the end of an episode - there is nothing left to promise."""
    return actions[start:start + horizon]


def side(actions):
    """Which tomato this episode went for. The arm swings left or right, and
    that is the ONE decision the first two seconds of every episode hide."""
    return 'left' if actions[:, 0].min() < -0.2 else 'right'
