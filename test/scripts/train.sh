#!/usr/bin/env bash
set -euox pipefail

TRAIN="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train"
TEST="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
LLM="transformer://scbirlab/lchemme-base-zinc22-lteq300:clean_smiles~mean"

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/original
CACHE="$OUTPUT_DIR/cache"
OUTPUT="$OUTPUT_DIR/models"
HYPERPARAMS="$OUTPUT_DIR"/hyperopt.json

START=50
STOP=550

printf \
    '{"n_units": [8, 16], "n_hidden": [4, 6], "residual_depth": [null, 2]}' \
| duvidnn hyperprep \
    -o "$HYPERPARAMS"

for class in chemprop fingerprint #chemprop
do
    for i in 0 1
    do
        HF_HOME="$CACHE" duvidnn train \
            -1 "$TEST" \
            -2 "$TRAIN" \
            -x clogp \
            -S smiles \
            -y log_rlm \
            -c "$HYPERPARAMS" \
            -k "$class" \
            -i $i \
            --output "$OUTPUT/$class-$i" \
            --cache "$CACHE" \
            --epochs 2 \
            -z 10 \
            --learning-rate 0.001 \
            --descriptors \
            --fp
        ls -lah "$OUTPUT"
        ls -lah "$OUTPUT"/"$class-$i"/*
        outfile="$script_dir"/outputs/predictions/"$class-$i.csv"
        HF_HOME="$CACHE" duvidnn predict \
            --test "$TRAIN" \
            --checkpoint "$OUTPUT"/"$class-$i" \
            --start $START \
            --end $STOP \
            --variance \
            --tanimoto \
            --doubtscore \
            --information-sensitivity \
            --last-layer \
            --optimality \
            -y log_rlm \
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
