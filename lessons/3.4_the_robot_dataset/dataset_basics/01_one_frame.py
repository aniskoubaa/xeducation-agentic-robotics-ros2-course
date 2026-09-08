"""Lesson 3.2 - the four ingredients, in one real frame.

A robot dataset is not mysterious.  Open ONE frame and you find exactly four
things: what the robot saw, where it was, what it did next, and what it was
told to do.  Everything else in the folder is bookkeeping around these four.

    python3 01_one_frame.py            # frame 33 - the moment the gripper closes
    python3 01_one_frame.py 100        # any other frame
"""

import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

REPO_ID = 'raiseschool/raise_ripeness_sort_ref'
DATASET = 'dataset'                  # the folder next door - 49 real pick demos
I = int(sys.argv[1]) if len(sys.argv) > 1 else 33

ds = LeRobotDataset(REPO_ID, root=DATASET)
f = ds[I]                            # ONE frame = one training example

def seven(t):                        # 7 floats, one tidy line
    return '[' + ' '.join(f'{x:+.3f}' for x in t.tolist()) + ']'

print(f'frame {I} of {ds.num_frames}   episode {int(f["episode_index"])}'
      f'   step {int(f["frame_index"])}   t = {float(f["timestamp"]):.1f} s')

# ---- 1. SENSE - what the wrist camera saw at this instant ------------------
img = f['observation.images.wrist']
print('1 sense   observation.images.wrist', tuple(img.shape),
      '  224x224x3 on disk, CxHxW for the model')

# ---- 2. STATE - where the arm was, right NOW ------------------------------
print('2 state   observation.state       ', seven(f['observation.state']),
      ' 6 joints + gripper,  NOW')

# ---- 3. ACTION - what was commanded NEXT.  This one is the label! ---------
print('3 action  action                  ', seven(f['action']),
      ' the same 7,     but NEXT')

# ---- 4. INSTRUCTION - what it was told to do, in words --------------------
print('4 task    task                    ', repr(f['task']))

png = f'out/frame_{I:05d}.png'
Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype('uint8')).save(png)
print('wrote', png, ' <- the image really is in the dataset, not beside it')
