# action_chunks_demo — how a VLA produces actions

Lesson 3.7 showed that an action is just numbers. This one asks the practical
question that follows: **how many of those actions does the model commit to at
once, and how does it build them?** Both halves are checked against the same
reference set (`raiseschool/raise_ripeness_sort_ref`, 49 picks at 10 Hz).

**No model is loaded and no GPU is touched.** Lesson 3.10 picks a VLA and 3.11
runs one. The two chunk sizes quoted here are read from LeRobot's own config
classes, not invented.

| file | it shows |
|---|---|
| `chunks.py` | a chunk is a slice: `actions[t : t+H]`, plus which side each episode went for |
| `01_one_action_or_a_chunk.py` | one action `(7,)` against a chunk `(50, 7)` — 0.1 s or 5.0 s of promise |
| `02_how_many_model_calls.py` | the call budget: 97 forward passes, or 4 |
| `03_where_the_plans_diverge.py` | **how long a chunk is too long, measured** — plus a figure |
| `04_the_rhythm.py` | look → plan 50 → fly 25 → look again, as a schedule |
| `05_step_by_step.py` | autoregressive: why a per-step error piles up |
| `06_refine_the_whole_plan.py` | diffusion: noise over the whole chunk, cleaned up in passes |

## Run them

```bash
ln -sfn ../../../datasets/raiseschool__raise_ripeness_sort_ref dataset
mkdir -p out

~/raise_venvs/lerobot/bin/python3 01_one_action_or_a_chunk.py
# ... 02 through 06 the same way
```

`dataset/` and `out/` are gitignored.

## Three things worth knowing

**The trade-off is visible in this dataset.** 25 episodes fetch the tomato on
the right and 24 on the left, and for the first **2.1 seconds every one of them
does exactly the same thing** — unfold from home, nothing decided. Only then
does the arm turn toward one fruit or the other. A chunk of 50 promises 5.0 s
of motion, so **2.9 s of it lands past the moment the decision is made**. That
is precisely why you execute part of a chunk and re-plan. `out/divergence.png`
draws all 49 traces overlapping and then forking.

**`05` and `06` are not models.** `05` pushes a fixed 0.20° per-step slip
through the two arrangements to show that the autoregressive one inherits every
earlier mistake (a random walk: 0.20° → 1.44° over 50 steps) while a whole-chunk
prediction does not (0.20° throughout). `06` starts from noise across all 50
actions and moves toward the real plan to show the *shape* of a diffusion pass.
A trained model has to **learn** what to clean off; here we simply move toward
the answer. SmolVLA uses flow matching, which is of this family.

**The numbers `50` and `100` are the libraries'.** `02` constructs
`SmolVLAConfig()` and `ACTConfig()` and prints `chunk_size` from them — 50 and
100 respectively. Constructing a config loads no weights.
