#!/usr/bin/env bash

set -e
set -x

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"
TEST="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

START=50
STOP=550

printf \
    '{"n_units": [8, 16], "n_hidden": [4, 6], "residual_depth": [null, 2]}' \
| duvidnn hyperprep \
    -o test/outputs/hyperopt.json

for class in fingerprint chemprop
do
    for i in 0 1
    do
        duvidnn train \
            -1 "$TEST" \
            -2 "$TRAIN" \
            -S smiles \
            -y clogp \
            -c test/outputs/hyperopt.json \
            -k "$class" \
            -i $i \
            --output "test/outputs/models/$class-$i" \
            --cache test/outputs/cache \
            --epochs 2 \
            -z 10 \
            --learning-rate 1e-5 \
            --descriptors \
            --fp
        ls -lah test/outputs/models
        ls -lah test/outputs/models/"$class-$i"/*
        outfile="test/outputs/predictions/$class-$i.csv"
        duvidnn predict \
            --test "$TRAIN" \
            --checkpoint test/outputs/models/"$class-$i" \
            --start $START \
            --end $STOP \
            --variance \
            --tanimoto \
            --doubtscore \
            --optimality \
            -y clogp \
            --cache test/outputs/cache \
            --output "$outfile"
        if [ ! "$(cat "$outfile" | wc -l)" -eq $(( $STOP - $START + 1 )) ]
        then
            echo "Predictions have wrong number of rows: $(cat "$outfile" | wc -l)"
            exit 1
        else
            cat "$outfile" | tr , $'\t' | head -n50
        fi
    done
done