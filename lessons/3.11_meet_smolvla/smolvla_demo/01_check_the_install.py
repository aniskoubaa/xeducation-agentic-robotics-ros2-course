"""Lesson 3.11 - is SmolVLA actually installed?

WHAT  Ask the environment four questions: which LeRobot, which torch, is there
      a GPU, and is the smolvla policy registered.

WHY   SmolVLA is not a package of its own - it ships inside LeRobot as one
      policy among a dozen.  If `lerobot` imports and `smolvla` is in the
      registry, you have everything.

    python3 01_check_the_install.py
"""

import shutil

import lerobot
import torch
from lerobot.policies.factory import make_policy_config
from smol import banner

banner('The install')
print(f'lerobot      {lerobot.__version__}')
print(f'torch        {torch.__version__}')
print(f'GPU          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none - CPU only"}')

# SmolVLA is a *policy type* inside LeRobot, not a separate import.
cfg = make_policy_config('smolvla')
print(f'policy       {cfg.type} is registered, chunk_size {cfg.chunk_size}')
print(f'trainer      {"lerobot-train" if shutil.which("lerobot-train") else "MISSING"} is on the PATH')
