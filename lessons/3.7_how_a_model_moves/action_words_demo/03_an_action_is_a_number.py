"""Lesson 3.7 - step 3: encode one real action as seven action words.

WHAT  Take the exact frame where our episode closes the gripper, and turn its
      seven joint angles into seven integers from the 256-word vocabulary.

WHY   This is the whole trick of slide 3.  A word became an index; a joint
      angle becomes an index in exactly the same way.  Once every action is a
      small integer, "predict the next action" is the same SHAPE of problem as
      "predict the next word" - so a model that can do one can do the other.

    python3 03_an_action_is_a_number.py
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from codebook import NAMES, all_actions, build, encode

EPISODE, STEP = 1, 54                    # the frame where this episode grasps

low, high = build(all_actions())
ds = LeRobotDataset('raiseschool/raise_ripeness_sort_ref', root='dataset')
action = ds[ds.meta.episodes[EPISODE]['dataset_from_index'] + STEP]['action'].numpy()
words = encode(action, low, high)

print(f'episode {EPISODE}, t = {STEP / ds.fps:.1f} s - the moment it closes the gripper')
for name, angle, word in zip(NAMES, action, words):
    print(f'   {name:<14s} {angle:+7.3f} rad  ->  action word {word:3d}')
print(f'the model never sees the angles: it sees {words.tolist()}')
