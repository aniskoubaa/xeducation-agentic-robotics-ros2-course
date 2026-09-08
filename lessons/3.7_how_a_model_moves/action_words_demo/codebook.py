"""The action vocabulary - build it once, use it in every script here.

OpenVLA's recipe, in full: take each joint separately, find the range that
covers the middle 98% of the training data, and cut that range into 256 steps.
An action word is then just which step you are on - an integer 0..255, exactly
like a word's index in an LLM vocabulary.

Percentiles rather than min/max, so a single freak frame cannot stretch the
scale and waste 200 of the 256 bins on values that never occur.
"""

import numpy as np
import pyarrow.parquet as pq

BINS = 256                               # OpenVLA's number: one byte per joint
TABLE = 'dataset/data/chunk-000/file-000.parquet'
NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2',
         'wrist_3', 'gripper']


def all_actions():
    """Every action in the dataset, as a 4228 x 7 array (no images decoded)."""
    table = pq.read_table(TABLE, columns=['action'])
    return np.stack(table['action'].to_numpy(zero_copy_only=False))


def build(actions):
    """The low and high edge of each joint's range: the middle 98% of the data."""
    return np.percentile(actions, 1, axis=0), np.percentile(actions, 99, axis=0)


def encode(action, low, high):
    """Joint angles -> action words. Anything past an edge is pinned to it."""
    step = (action - low) / (high - low) * (BINS - 1)
    return np.clip(np.round(step), 0, BINS - 1).astype(int)


def decode(words, low, high):
    """Action words -> joint angles: the angle at the centre of that bin."""
    return low + words / (BINS - 1) * (high - low)
