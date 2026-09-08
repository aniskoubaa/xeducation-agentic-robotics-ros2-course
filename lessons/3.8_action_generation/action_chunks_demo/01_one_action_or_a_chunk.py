"""Lesson 3.8 - step 1: one action, or the next fifty?

WHAT  Cut both out of the same real episode and print their shapes: the single
      action a step-at-a-time policy commits to, and the chunk a modern one does.

WHY   Slide 2's left side against its right side, on real numbers.  A chunk is
      not a new kind of object - it is H consecutive rows of the table we have
      been reading since lesson 3.4.  What changes is how much the robot
      promises to do before it looks at the world again: 0.1 s, or 5 seconds.

    python3 01_one_action_or_a_chunk.py
"""

from chunks import FPS, chunk, episodes

EPISODE, START, HORIZON = 1, 0, 50       # H = 50 is SmolVLA's default chunk

actions = episodes()[EPISODE]
one = actions[START]
plan = chunk(actions, START, HORIZON)

print(f'episode {EPISODE}: {len(actions)} actions, {len(actions) / FPS:.1f} s of one pick')
print()
print(f'  ONE ACTION    shape {str(one.shape):<9s} {1 / FPS:.1f} s of motion, then look again')
print(f'                {" ".join(f"{v:+.2f}" for v in one)}')
print()
print(f'  A CHUNK       shape {str(plan.shape):<9s} {len(plan) / FPS:.1f} s of motion, promised in one go')
print(f'    first       {" ".join(f"{v:+.2f}" for v in plan[0])}')
print(f'    last        {" ".join(f"{v:+.2f}" for v in plan[-1])}')
print()
print('the chunk is not a new object: it is the next H rows of the same table.')
