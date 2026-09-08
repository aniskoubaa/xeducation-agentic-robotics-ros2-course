"""Shared setup for the three SmolVLA demos.

Two model names and one instruction, in one place, so every script below is
short enough to read on a slide.  The rest is noise control: loading a 450M
model prints device chatter and download bars that would bury the output.
"""

import logging
import warnings

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

BASE = 'lerobot/smolvla_base'            # the 450M checkpoint from LeRobot
REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
TASK = 'pick the red tomato'             # the one instruction in our dataset
EPISODE, START = 0, 25                   # the frame where the tomato fills the camera
HORIZON = 50                             # SmolVLA's chunk size: 50 actions, 5 s at 10 Hz


def one_frame(dataset):
    """The single observation every demo here feeds the model."""
    return dataset[dataset.meta.episodes[EPISODE]['dataset_from_index'] + START]


def banner(text):
    print(text)
    print('-' * len(text))
