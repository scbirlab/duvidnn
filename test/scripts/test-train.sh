#!/usr/bin/env bash
set -euox pipefail

script_dir=$(readlink -f $(dirname "$0"))
OUTPUT_DIR=$(readlink -f "$script_dir"/..)/outputs/config-cli
CACHE="$OUTPUT_DIR/cache"
CHECKPOINT="$OUTPUT_DIR/model"
CONFIG="$OUTPUT_DIR/config.json"
TRAIN="$OUTPUT_DIR/training.parquet"
PREDICT="$OUTPUT_DIR/prediction.parquet"
PREDICTIONS="$OUTPUT_DIR/predictions.parquet"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

python - <<EOF
import json

import pandas as pd


pd.DataFrame({
    "x": [
        [0.],
        [1.],
        [2.],
        [3.],
    ],
    "y": [
        [0.],
        [2.],
        [4.],
        [6.],
    ],
}).to_parquet(
    "$TRAIN",
    index=False,
)

pd.DataFrame({
    "x": [
        [4.],
        [5.],
    ],
}).to_parquet(
    "$PREDICT",
    index=False,
)

config = {
    "box": {
        "model": {
            "class_path": "torch.nn.Linear",
            "init_args": {
                "in_features": 1,
                "out_features": 1,
            },
        },
        "pipeline": {},
        "input_map": {
            "inputs": {
                "input": "x",
            },
            "target": "y",
        },
    },
    "trainer": {
        "max_epochs": 2,
        "loss": {
            "class_path": "torch.nn.MSELoss",
        },
        "optimizer": "torch.optim.Adam",
        "optimizer_kwargs": {
            "lr": 0.01,
        },
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
    },
    "fit": {
        "batch_size": 2,
    },
}

with open(
    "$CONFIG",
    "w",
) as f:
    json.dump(
        config,
        f,
        indent=4,
    )
EOF

duvidnn train \
    --config "$CONFIG" \
    --training "$TRAIN" \
    --output "$CHECKPOINT" \
    --cache "$CACHE" \
    --set trainer.max_epochs=1 \
    --set fit.batch_size=2

test -f "$CHECKPOINT/config.json"
test -d "$CACHE"
test -d "$CACHE/datasets"
test -d "$CACHE/hub"

duvidnn predict \
    --checkpoint "$CHECKPOINT" \
    --data "$PREDICT" \
    --output "$PREDICTIONS" \
    --cache "$CACHE" \
    --set batch_size=1

test -f "$PREDICTIONS"

python - <<EOF
import pandas as pd


predictions = pd.read_parquet(
    "$PREDICTIONS"
)

assert len(predictions) == 2, (
    "Expected 2 predictions, "
    f"but found {len(predictions)}."
)

assert "prediction" in predictions.columns, (
    "Prediction output is missing "
    "the 'prediction' column."
)

print(predictions)
EOF

if [ -e "$CACHE/datasets/datasets" ]
then
    echo "Nested datasets cache detected: $CACHE/datasets/datasets"
    exit 1
fi

echo "CLI config/cache smoke test passed."
