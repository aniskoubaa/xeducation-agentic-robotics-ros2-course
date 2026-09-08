"""Lesson 3.7 - step 5: the sentence and the movement, in the same alphabet.

WHAT  Print the instruction as the language model sees it, and the robot's
      action at the same moment as a VLA sees it, in the same layout.

WHY   This is the sentence to carry out of the lesson: predicting an action is
      just like predicting a word.  Above, four integers drawn from a 151,643
      word vocabulary.  Below, seven integers drawn from a 256 word one.  Both
      are sequences of indices, so the same machinery predicts either.

    python3 05_side_by_side.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from transformers import AutoTokenizer

from codebook import BINS, all_actions, build, encode

EPISODE, STEP = 1, 54
SHORT = ['pan', 'lift', 'elbow', 'wr_1', 'wr_2', 'wr_3', 'grip']

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
ids = tokenizer.encode('pick the red tomato', add_special_tokens=False)
pieces = [w.replace(chr(288), '_') for w in tokenizer.convert_ids_to_tokens(ids)]

low, high = build(all_actions())
ds = LeRobotDataset('raiseschool/raise_ripeness_sort_ref', root='dataset')
action = ds[ds.meta.episodes[EPISODE]['dataset_from_index'] + STEP]['action'].numpy()
words = encode(action, low, high)

print(f'WORDS    "pick the red tomato"      vocabulary {tokenizer.vocab_size:,}')
print('        ' + ''.join(f'{p:>9s}' for p in pieces))
print('        ' + ''.join(f'{i:>9d}' for i in ids))
print()
print(f'ACTION   the same robot, t = {STEP / ds.fps:.1f} s      vocabulary {BINS}')
print('        ' + ''.join(f'{s:>9s}' for s in SHORT))
print('        ' + ''.join(f'{w:>9d}' for w in words))
print()
print('both are a sequence of indices into a fixed vocabulary - so the same model can predict either')
