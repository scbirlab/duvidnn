#!/usr/bin/env bash

set -e
set -x

printf \
    '{"n_units": [16, 32], "n_hidden": [1, 2, 3]}' \
| duvida hyperprep \
    -o test/outputs/hyperopt.json

for class in fingerprint chemprop
do
    duvida train \
        -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
        -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
        -S smiles \
        -x "transformers://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean" \
        -y clogp \
        -c test/outputs/hyperopt.json \
        -k "$class" \
        -i 0 \
        --prefix test/outputs/models \
        --cache test/outputs/cache \
        --epochs 10 \
        --learning-rate 1e-5 \
        --descriptors \
        --fp
done