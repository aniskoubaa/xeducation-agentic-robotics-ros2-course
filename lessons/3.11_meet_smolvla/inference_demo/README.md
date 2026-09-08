# Lesson 3.11 — Demo 3 of 3: Run the Fine-Tuned Policy

Load the checkpoint demo 2 produced and ask it to act. `model/` points at
`runs/ripeness/checkpoints/last/pretrained_model` — the only folder inference
needs.

| file | what it shows |
|---|---|
| `smol.py` | the shared names, plus `quiet()` — SmolVLA chatters as it loads |
| `01_load_the_policy.py` | the three lines of an inference setup, and the contract they load |
| `02_ask_it_to_act.py` | one observation in, 50 actions out, beside what the human did |
| `03_the_same_picture.py` | demo 1's figure redrawn — the before/after of fine-tuning |
| `04_a_whole_episode.py` | 97 steps driven by 4 model calls, lesson 3.8's rhythm for real |
| `05_fast_enough.py` | milliseconds per call against a 10 Hz robot's clock |

## Run them

```bash
python3 01_load_the_policy.py
python3 02_ask_it_to_act.py
python3 03_the_same_picture.py     # writes out/tuned_plan.png
python3 04_a_whole_episode.py      # writes out/rollout.png
python3 05_fast_enough.py
```

## Inference, in full

```python
policy = SmolVLAPolicy.from_pretrained('model')
pre, post = make_pre_post_processors(policy.config, 'model')

chunk = post(policy.predict_action_chunk(pre({
    'observation.images.wrist': image,      # (3, 224, 224) float
    'observation.state': joints,            # (7,) radians
    'task': 'pick the red tomato'})))       # -> (1, 50, 7)
```

The preprocessor renames `wrist` to `camera1`, tokenises the instruction, moves
everything to the GPU and normalises it. The postprocessor un-normalises the
actions and brings them back to the CPU. Both were saved by training, so the
numbers that come out are already in radians.

## What you should take away

* Same 450M parameters as demo 1, same frame, same sentence. Before fine-tuning
  the plan was **~40 degrees per joint** away from the human's; after, it is
  **about a fifth of a degree** — and the gripper, which the base model did not even
  have, opens and closes at the right moments.
* One call plans 50 actions; we execute 25 and look again. **Four calls cover a
  whole 9.7 s episode**, at well under a fifth of the robot's clock.

## What this is *not*

`04_a_whole_episode.py` feeds the policy the **recorded** observations, not the
ones a robot would have seen after following the policy's own actions. It shows
that the policy reproduces the demonstrations; it does not show that the robot
picks the tomato. Closing that loop, and measuring success rather than error, is
lesson 3.12.
