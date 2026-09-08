# Lesson 3.11 — Demo 2 of 3: Fine-Tune SmolVLA on Our Data

One `lerobot-train` command, our own 49 teleoperated episodes, one laptop GPU.
The run in the video is a short live one you can sit through; the policy demo 3
loads came from the same script with `STEPS=20000`.

| file | what it shows |
|---|---|
| `smol.py` | the names shared by all three demos, and the noise control |
| `01_what_the_trainer_sees.py` | our dataset vs what `smolvla_base` declares it expects |
| `02_what_actually_trains.py` | only 99,880,992 of 450,046,176 numbers are free to move |
| `finetune.sh` | the fine-tuning command itself |
| `03_the_loss_curve.py` | the real 20,000-step log, plotted |
| `04_what_came_out.py` | the checkpoints on disk, and how the config changed |

## Run it

```bash
python3 01_what_the_trainer_sees.py
python3 02_what_actually_trains.py

bash finetune.sh                       # 300 steps, about a minute — watch the loss

# the policy demo 3 uses, ~70 minutes on one RTX 4090 laptop GPU:
STEPS=20000 LOG=100 SAVE=5000 OUT=runs/ripeness bash finetune.sh

python3 03_the_loss_curve.py           # writes out/loss_curve.png
python3 04_what_came_out.py
```

To watch only the lines that matter while it trains (LeRobot draws a progress
bar with carriage returns, so turn those into newlines first):

```bash
bash finetune.sh 2>&1 | stdbuf -oL tr '\r' '\n' \
  | grep --line-buffered -oE '(Creating |step:).*'
```

## The two flags that are not obvious

* `--rename_map='{"observation.images.wrist": "observation.images.camera1"}'` —
  `smolvla_base` declares three cameras called `camera1..3`; our recording has
  one called `wrist`. Without this the run dies at startup with a feature
  mismatch. It is the single most common first failure.
* `--policy.push_to_hub=false` — otherwise LeRobot insists on a `policy.repo_id`
  to publish to.

## What you should take away

* **99,880,992 of 450,046,176 parameters train** (22%). `train_expert_only` is
  `True` by default: the vision encoder, the connector and the whole language
  model stay frozen, and only the action expert plus the small robot-facing
  heads learn. That is why 49 episodes and **under 4 GB of GPU memory** are enough.
* Fine-tuning rewrites the *contract*, not only the weights: the checkpoint that
  comes out declares **7 action numbers** (the gripper is back), knows our camera
  by name, and carries the mean and spread of **our** 4228 frames.
* The loss falls steeply through the 1,000-step warm-up, then settles. Flat and
  low is what you are waiting for — but a low training loss is not evidence the
  robot succeeds. That question is lesson 3.12.

`runs/ripeness_20000.log` is the real log of the shipped run, one line per 100
steps, exactly as `finetune.sh` printed it.
