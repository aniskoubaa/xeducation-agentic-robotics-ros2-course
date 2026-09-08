"""Lesson 3.11 - pull the SmolVLA weights down from the Hub.

WHAT  Download (or find, already cached) the `lerobot/smolvla_base` checkpoint
      and name the four things a SmolVLA checkpoint is actually made of.

WHY   A checkpoint is not one opaque blob.  It is the weights, plus a config
      saying what shape of robot they expect, plus two processors that
      normalise what goes in and un-normalise what comes out.  All four get
      copied and re-fitted when you fine-tune - so it is worth knowing them.

    python3 02_get_the_weights.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from smol import BASE, banner

# What each file is for.  Anything not listed here is documentation.
ROLE = {'model.safetensors': 'the weights - every number in the model',
        'config.json': 'the shape of robot and task it expects',
        'policy_preprocessor.json': 'how an observation is normalised on the way in',
        'policy_postprocessor.json': 'how an action is un-normalised on the way out'}

def size(path):
    kb = path.stat().st_size / 1000
    return f'{kb / 1000:.0f} MB' if kb > 1000 else f'{kb:.1f} KB'


banner(f'{BASE}')

folder = Path(snapshot_download(BASE))       # downloads once, then reads the cache
for name, role in ROLE.items():
    print(f'  {name:28s} {size(folder / name):>8s}   {role}')

stats = sorted(folder.glob('*normalizer*.safetensors'))
print(f'  {f"+ {len(stats)} statistics files":28s} {"":8s}   the mean and spread it normalises against')
print(f'\n  {size(folder / "model.safetensors")} on disk for 450M parameters - it fits on one GPU, which is the whole point.')
