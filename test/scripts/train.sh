#!/usr/bin/env bash

set -e
set -x

printf \
    '{"n_units": [8, 16], "n_hidden": [4, 6], "residual_depth": [null, 2]}' \
| duvida hyperprep \
    -o test/outputs/hyperopt.json

for class in fingerprint chemprop
do
    for i in 0 1
    do
        duvida train \
            -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
            -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
            -S smiles \
            -x "transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean" \
            -y clogp \
            -c test/outputs/hyperopt.json \
            -k "$class" \
            -i $i \
            --prefix test/outputs/models \
            --cache test/outputs/cache \
            --epochs 2 \
            --learning-rate 1e-5 \
            --descriptors \
            --fp
        done
done

for class in fingerprint chemprop
do
    duvida train \
        -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
        -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
        -S smiles \
        -x "transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean" \
        -y clogp \
        -c test/outputs/hyperopt.json \
        -k "$class" \
        -i 1 \
        --prefix test/outputs/models \
        --cache test/outputs/cache \
        --epochs 10 \
        --learning-rate 1e-5 \
        --descriptors \
        --fp
done