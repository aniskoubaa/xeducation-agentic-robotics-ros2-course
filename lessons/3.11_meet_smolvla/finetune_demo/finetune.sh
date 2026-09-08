#!/usr/bin/env bash
# Fine-tune SmolVLA on our own 49 teleoperated episodes.  One GPU, no cloud.
# The shipped policy:  STEPS=20000 LOG=100 SAVE=5000 OUT=runs/ripeness bash finetune.sh
STEPS=${STEPS:-300}; LOG=${LOG:-60}; SAVE=${SAVE:-$STEPS}; OUT=${OUT:-runs/live}
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=raiseschool/raise_ripeness_sort_ref --dataset.root=dataset \
  --rename_map='{"observation.images.wrist": "observation.images.camera1"}' \
  --batch_size=16 --steps="$STEPS" --log_freq="$LOG" --save_freq="$SAVE" \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
  --num_workers=4 --seed=1000 --output_dir="$OUT"
