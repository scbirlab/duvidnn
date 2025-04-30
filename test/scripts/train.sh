#!/usr/bin/env bash

set -e
set -x

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"
TEST="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

printf \
    '{"n_units": [8, 16], "n_hidden": [4, 6], "residual_depth": [null, 2]}' \
| duvida hyperprep \
    -o test/outputs/hyperopt.json

for class in fingerprint chemprop
do
    for i in 0 1
    do
        duvida train \
            -1 "$TRAIN" \
            -2 "$TEST" \
            -S smiles \
            -y clogp \
            -c test/outputs/hyperopt.json \
            -k "$class" \
            -i $i \
            --prefix test/outputs/models \
            --output "$class-$i" \
            --cache test/outputs/cache \
            --epochs 2 \
            --learning-rate 1e-5 \
            --descriptors \
            --fp
        ls -lah test/outputs/models
        ls -lah test/outputs/models/"$class-$i"/*
        duvida predict \
            --test "$TEST" \
            --checkpoint test/outputs/models/"$class-$i" \
            --start 10 \
            --end 2_000 \
            --variance \
            --tanimoto \
            --doubtscore \
            --information-sensitivity \
            --optimality \
            -y clogp \
            --output test/outputs/predictions/"$class-$i".csv.gz
        if [ ! "$(zcat "$class-$i".csv.gz | wc -l)" -eq "1990" ]
        then
            echo "Predictions have wrong number of rows: $(zcat "$class-$i".csv.gz | wc -l)"
            exit 1
        else
            zcat "$class-$i".csv.gz | tr , $'\t' | head -n50
        fi
    done
done