# Lesson 3.11 — Demo 1 of 3: Meet SmolVLA

Install it, open it up, and run it once. No training here — the point is to see
what a 450M-parameter VLA is actually made of, and what its interface looks like.

Everything runs against the reference dataset
(`raiseschool/raise_ripeness_sort_ref`, 49 episodes) linked in as `dataset/`,
and against the real `lerobot/smolvla_base` checkpoint from the Hugging Face Hub.

| file | what it shows |
|---|---|
| `install.sh` | the two commands that put SmolVLA on a machine |
| `smol.py` | the names shared by all three demos, and the noise control |
| `01_check_the_install.py` | LeRobot, torch, the GPU, and `smolvla` in the policy registry |
| `02_get_the_weights.py` | the four things a SmolVLA checkpoint is made of |
| `03_whats_inside.py` | 450,046,176 parameters, counted part by part |
| `04_the_two_parts.py` | what the VLM half and the action-expert half are wired to do |
| `05_one_look_one_plan.py` | one real frame in, a chunk of 50 actions out |
| `06_what_a_plan_looks_like.py` | that chunk drawn against what the human actually did |

## Run them

```bash
source .venv/bin/activate          # see install.sh
python3 01_check_the_install.py
python3 02_get_the_weights.py
python3 03_whats_inside.py
python3 04_the_two_parts.py
python3 05_one_look_one_plan.py
python3 06_what_a_plan_looks_like.py     # writes out/base_plan.png
```

## What you should take away

* **450,046,176 parameters** — 350,165,184 in the vision-language model,
  98,245,840 in the action expert. The deck's "~350M + ~100M" is exact.
* The interface of a VLA is three things in and one thing out: **an image, a
  state, a sentence → a chunk of 50 future actions**, about half a second per call.
* The base model runs on our robot's data and produces a perfectly plausible
  plan **that is roughly 40 degrees per joint away from the right one**. It has
  never seen this arm or this task. Demo 2 fixes that.
* It also produces **six** numbers, not seven: `smolvla_base` was configured for
  a six-joint arm. Fine-tuning re-fits the model to our arm, gripper included.

> Flow matching starts from random noise, so two runs of script 05 or 06 give
> slightly different plans. That is the method, not a bug.
