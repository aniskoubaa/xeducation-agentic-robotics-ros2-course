"""Lesson 3.8 - step 2: what chunking actually buys - fewer model calls.

WHAT  Count how many times the network has to run to get through one episode,
      one action at a time and then with chunks of various lengths.

WHY   "Fewer expensive model calls" is the first reason slide 2 gives, and it
      is arithmetic, not opinion.  The same episode needs 97 forward passes at
      one action per pass, or 4 if the model plans 5 seconds and re-plans every
      2.5.  The last column is the price: how far ahead the robot has committed
      before it is allowed to look at the world again.

    python3 02_how_many_model_calls.py
"""

import logging
logging.disable(logging.WARNING)         # the config classes chatter about devices

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from chunks import FPS, episodes

EPISODE = 1
actions = episodes()[EPISODE]
steps = len(actions)

print('where H comes from - the libraries own defaults (configs only, no model loaded):')
for name, config in (('SmolVLA', SmolVLAConfig()), ('ACT', ACTConfig())):
    print(f'   {name:<9s} chunk_size {config.chunk_size:3d}  = {config.chunk_size / FPS:4.1f} s of motion per plan')
print()
print(f'episode {EPISODE} is {steps} actions ({steps / FPS:.1f} s).  how often must the network run?')
for horizon, execute in ((1, 1), (50, 50), (50, 25), (50, 10)):
    calls = -(-steps // execute)         # ceiling division: the last plan is partial
    how = 'one action at a time' if horizon == 1 else f'chunk {horizon}, re-plan every {execute}'
    print(f'   {how:<32s} {calls:3d} calls   commits {execute / FPS:4.1f} s ahead')
