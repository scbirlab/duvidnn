#!/usr/bin/env bash

set -e
set -x

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"

duvidnn percentiles \
    "$TRAIN" \
    --columns clogp tpsa \
    --percentiles 1 5 10 \
    --batch 128 \
    --cache test/outputs/cache \
    --output test/outputs/split/percentiles.csv \
    --plot test/outputs/split/percentiles-plot.png \
    --structure smiles

for type in faiss scaffold
do
    duvidnn split \
        "$TRAIN" \
        --train .7 \
        --validation .15 \
        -S smiles \
        --type $type \
        -k 2 \
        --seed 1 \
        --cache test/outputs/cache \
        --output test/outputs/split/$type.csv \
        --plot test/outputs/split/$type-plot.png
done