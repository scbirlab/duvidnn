#!/usr/bin/env bash

set -e
set -x

printf \
    '{"model_class": ["fingerprint"], "n_units": [16, 32], "n_hidden": [1, 2, 3]}' \
| duvida hyperprep \
    -o test/outputs/hyperopt.json

duvida train \
    hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    -x smiles \
    -y clogp \
    -c test/outputs/hyperopt.json \
    -i 0 \
    --prefix test/outputs/models \
    --cache test/outputs/cache \
    --epochs 2 \
    --learning-rate 1e-3 \
    --descriptors \
    --fp