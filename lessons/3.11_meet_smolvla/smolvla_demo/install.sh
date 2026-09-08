#!/usr/bin/env bash
# How SmolVLA gets onto your machine.  Run once; it pulls ~3 GB of wheels.

python3 -m venv .venv                 # 1. a clean environment
source .venv/bin/activate

pip install "lerobot[smolvla]"        # 2. LeRobot, plus the SmolVLA extra

# That is the whole install.  The 450M weights come down on first use.
