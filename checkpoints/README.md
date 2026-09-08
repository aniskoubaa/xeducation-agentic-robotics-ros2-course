# Model checkpoints — getting the trained brain

Trained SmolVLA checkpoints are **not stored in git**: the reference run is
1.3 GB. This page says how to get one. The *dataset* it was trained on **is** in
this repo (`datasets/`), so you can also reproduce the checkpoint yourself.

## The reference checkpoint

| | |
|---|---|
| Task | "pick the red tomato" among green distractors, at a plant row |
| Trained on | `datasets/raiseschool__raise_ripeness_sort_ref` — 49 episodes, in this repo |
| Recipe | SmolVLA base, 6000 steps, batch 64 — about 2 h on an RTX 4090 laptop (16 GB) |
| Expected local path | `~/raise_checkpoints/smolvla_C_ref/checkpoints/006000/pretrained_model` |

## Download it (about 2 minutes, no login)

The checkpoint is a public release asset (687 MB `tar.gz`):

<https://github.com/aniskoubaa/raise2026-student/releases/tag/day2-smolvla-ref-v3>

```bash
mkdir -p ~/raise_checkpoints/smolvla_C_ref/checkpoints/006000
curl -L -o /tmp/smolvla_C_ref.tar.gz \
  https://github.com/aniskoubaa/raise2026-student/releases/download/day2-smolvla-ref-v3/smolvla_C_ref.tar.gz
tar -xzf /tmp/smolvla_C_ref.tar.gz -C ~/raise_checkpoints/smolvla_C_ref/checkpoints/006000
```

The day-2 tools auto-detect the newest checkpoint under
`~/raise_checkpoints/smolvla_C_ref/checkpoints/` when `VLA_LOCAL_CKPT` is unset,
so once it is unpacked there is nothing to configure.

## Or train your own

Lesson 3.11's fine-tuning demo does exactly this, on exactly this dataset:
`lessons/3.11_meet_smolvla/finetune_demo/`. Expect a couple of hours on a
16 GB GPU. `raise_ros2_ws/src/raise2026_labs/day2/HOW_TO_TRAIN_AND_USE.md` is
the long-form version.
