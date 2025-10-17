#!/usr/bin/env bash

set -euox pipefail

TRAIN="hf://scbirlab/thomas-2018-spark-wt@Acinetobacter-baumannii:train"
TEST="hf://scbirlab/thomas-2018-spark-wt@Acinetobacter-baumannii:validation"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/strain
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/models"
HYPERPARAMS="$OUTPUT_DIR"/hyperopt.json

START=50
STOP=550

printf \
    '{"n_units": [8, 16], "n_hidden": [4, 6], "residual_depth": [null, 2]}' \
| duvidnn hyperprep \
    -o "$HYPERPARAMS"

for class in fingerprint chemprop
do
    for i in 0 1
    do
        XDG_CACHE_HOME="$CACHE" HF_DATASETS_CACHE="$CACHE" duvidnn train \
            -1 "$TEST" \
            -2 "$TRAIN" \
            -x species:vectome-fingerprint clogp mwt:log \
            -S smiles \
            -y pmic \
            -c "$HYPERPARAMS" \
            -k "$class" \
            -i $i \
            --output "$OUTPUT/$class-$i" \
            --cache "$CACHE" \
            --epochs 2 \
            -z 10 \
            --learning-rate 0.01 \
            --descriptors \
            --fp
        ls -lah "$OUTPUT"
        ls -lah "$OUTPUT"/"$class-$i"/*
        outfile="$OUTPUT_DIR"/predictions/"$class-$i.csv"
        XDG_CACHE_HOME="$CACHE" HF_DATASETS_CACHE="$CACHE" duvidnn predict \
            --test "$TRAIN" \
            --checkpoint "$OUTPUT"/"$class-$i" \
            --start $START \
            --end $STOP \
            --variance \
            --tanimoto \
            --doubtscore \
            --optimality \
            -y pmic \
            --cache "$CACHE" \
            --output "$outfile"
        output_nlines=$(cat "$outfile" | wc -l)
        if [ ! "$output_nlines" -eq "$(( $STOP - $START + 1 ))" ]
        then
            echo "Predictions have wrong number of rows: $output_nlines"
            exit 1
        else
            head -n50 "$outfile" | tr , $'\t'
        fi
    done
done
