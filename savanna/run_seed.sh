#!/bin/bash
# One seed of the acting stages on a GPU, from this folder (with base_weights.pt, tokenizer.json, training/ and the
# seed's stage-1 weights at runs/s1_seedS/weights_4000.pt): stage 2, then stage 3 with the stage-2 tests beside it,
# then the stage-3 tests. Each step logs to its own file.
set -u
cd "$(dirname "$0")"
s=${1:-0}
python3 -u train_act.py runs/s2_seed$s --init runs/s1_seed$s/weights_4000.pt --stage 2 --updates 1000 \
    --eval-at 250,500,1000 --seed $s --beta 0.5 > s2_seed$s.out 2>&1 || { echo "stage 2 failed"; exit 1; }
python3 -u train_act.py runs/s3_seed$s --init runs/s2_seed$s/weights_1000.pt --ref runs/s2_seed$s/weights_1000.pt \
    --stage 3 --updates 1500 --eval-at 250,500,1000,1500 --seed $s --beta 0.5 > s3_seed$s.out 2>&1 &
python3 -u battery_i1.py runs/s1_seed$s/weights_4000.pt runs/s2_seed$s/weights_1000.pt \
    --actor runs/s2_seed$s/weights_1000.pt --days 2500 --out results/s2_seed$s > bat_s2_seed$s.out 2>&1
wait
python3 -u battery_i1.py runs/s3_seed$s --actor runs/s3_seed$s/weights_1500.pt --days 2500 \
    --out results/s3_seed$s > bat_s3_seed$s.out 2>&1 &
python3 -u seek_test.py --models stage2=runs/s2_seed$s/weights_1000.pt stage3=runs/s3_seed$s/weights_1500.pt \
    s3_at_500=runs/s3_seed$s/weights_500.pt s3_at_1000=runs/s3_seed$s/weights_1000.pt --days 2000 \
    --out results/seek_seed$s > seek_seed$s.out 2>&1
wait
echo "all done"
