"""Shared setup for the evaluation demo.

One policy, one dataset, one split, in one place, so the scripts below stay
short enough to read on a slide.  The rest is noise control: loading a 450M
model prints device chatter and progress bars that would bury the numbers.
"""

import contextlib
import logging
import os
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

BASE = 'lerobot/smolvla_base'             # the 450M checkpoint, before any of our data
MODEL = 'model'                           # ours, trained on episodes 0-39 only
MODEL_ALL = 'model_all'                   # lesson 3.11's, trained on all 49
REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
TASK = 'pick the red tomato'

TRAIN = range(0, 40)                      # what the policy under test was shown
HELD_OUT = range(40, 49)                  # what it has never seen
SEEN = range(0, 5)                        # five training episodes, for the comparison
STRIDE = 5                                # evaluate twice a second, not ten times
HORIZON = 50                              # SmolVLA plans 50 actions at a time: 5 s at 10 Hz

HUMAN = 0.35                              # deg: how far apart two demonstrations of the
                                          # same scene are - measured by 02_the_yardstick.py

JOINTS = ['pan', 'lift', 'elbow', 'wrist1', 'wrist2', 'wrist3', 'gripper']
ARM = slice(0, 6)                         # the six arm joints; the gripper is its own story
EVAL = 'out/eval.npz'


def demonstrations():
    """Every action a human ever commanded, split by episode: {index: (T, 7)}."""
    import pathlib

    import numpy as np
    import pandas as pd

    files = sorted(pathlib.Path('dataset/data').rglob('*.parquet'))
    table = pd.concat(pd.read_parquet(f, columns=['episode_index', 'action']) for f in files)
    return {int(index): np.stack(rows.to_numpy())
            for index, rows in table.groupby('episode_index')['action']}


@contextlib.contextmanager
def quiet():
    """SmolVLA prints three lines about SmolVLM2 every time it loads its VLM."""
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        yield


def banner(text):
    print(text)
    print('-' * len(text))
