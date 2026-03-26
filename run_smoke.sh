#!/usr/bin/env bash
set -e
echo "Running Unit Tests..."
python -m pytest tests/ -q

echo "Running Fast Training..."
# Running train.py. We need to ensure it doesn't run forever.
# We'll rely on BootstrappedSharpeCallback or just force max_epochs=1 via code (but train.py has 30 hardcoded).
# The prompt says "python train.py --fast_dev_run true"
# My train.py needs to verify if it accepts CLI args.
# Currently my train.py main() hardcodes max_epochs=30.
# I should update train.py to accept basic CLI overrides or use LightningCLI.
# But I am not asked to refactor train.py unless "Make it executable" implies modifying it to work.
# Wait, prompt: "python train.py --fast_dev_run true --data toy_data/"
# My train.py does NOT parse these args currently.
# I MUST update train.py to handle this, otherwise smoke test checks nothing or fails.
# I will patch train.py to use `LightningCLI` or `argparse`.
# Given constraints, I'll add simple argparse to train.py in a separate step or just rely on hardcoded for now
# and let the user know, OR I proactively fix it now.
# Prompt says "python train.py --fast_dev_run true". I should support it.
# BUT fast_dev_run disables checkpointing. We need a checkpoint for roll.py.
# So we run minimal epochs instead.
python train.py --data toy_data/ --max_epochs 2

echo "Running Roll..."
# Need a checkpoint. train.py should have produced one in lightning_logs/version_X/checkpoints/
# We find the latest one.
CKPT=$(find lightning_logs -name "*.ckpt" | head -n 1)
if [ -z "$CKPT" ]; then
  echo "No checkpoint found!"
  exit 1
fi

python roll.py --ckpt "$CKPT" --data toy_data/ --out weights_smoke.csv --date 2023-12-31

echo "SMOKE PASS"
