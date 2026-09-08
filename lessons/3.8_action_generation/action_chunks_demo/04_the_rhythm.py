"""Lesson 3.8 - step 4: the loop to memorise - look, plan, execute part, re-plan.

WHAT  Walk one real episode through the chunking loop and print the schedule:
      when the model runs, how much it promises, and how much gets flown before
      it looks again.

WHY   The bottom of slide 2: sense, predict a chunk, execute PART of it,
      re-plan, repeat.  Executing only part is the whole point - it keeps the
      benefit of a long plan while keeping the robot's information fresh.  The
      last column is the honest cost of the arrangement: how stale the plan the
      robot is currently flying has been allowed to become.

    python3 04_the_rhythm.py
"""

from chunks import FPS, chunk, episodes

EPISODE, HORIZON, EXECUTE = 1, 50, 25    # plan 5.0 s, fly 2.5 s, then look again

actions = episodes()[EPISODE]
print(f'episode {EPISODE}: {len(actions)} actions.  plan {HORIZON}, fly {EXECUTE}, look again:')
calls = 0
for start in range(0, len(actions), EXECUTE):
    plan = chunk(actions, start, HORIZON)
    flown = min(EXECUTE, len(actions) - start)
    calls += 1
    print(f'   t = {start / FPS:4.1f} s   LOOK -> plan {len(plan):2d} actions '
          f'({len(plan) / FPS:.1f} s)   fly {flown:2d} of them '
          f'({start / FPS:4.1f} - {(start + flown) / FPS:4.1f} s)')
print(f'{calls} model calls for {len(actions) / FPS:.1f} s of motion, and the plan being '
      f'flown is never more than {EXECUTE / FPS:.1f} s old.')
