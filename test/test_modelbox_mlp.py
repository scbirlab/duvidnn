"""Integration tests for TorchMLPModelBox.

Tests cover: model creation, training, prediction, checkpointing, evaluation.
No chemistry dependencies — always runs in CI.

Anti-self-fulfilling design:
- Shapes are asserted against known values from the fixture, not re-derived from the model.
- Checkpoint roundtrip asserts predictions are byte-for-byte identical, not just close.
- Weight-change test captures the initial state_dict before training starts.
- Aggregator output shapes are verified against independently-known dimensions.
"""

import json
import os

import numpy as np
import pytest

from duvidnn.torch.modelbox.modelboxes import TorchMLPModelBox

FEATURES = ["f1", "f2", "f3", "f4"]
LABEL = "y"
N_ENSEMBLE = 2
N_FEATURES = 4
N_TRAIN = 50
N_VAL = 10


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------

def test_create_model_dims():
    """Model built with explicit shapes has the correct input/output dimensions."""
    mb = TorchMLPModelBox(ensemble_size=N_ENSEMBLE, n_units=8, n_hidden=1)
    mb.input_shape = (N_FEATURES,)
    mb.output_shape = (1,)
    mb.model = mb.create_model()
    # n_input and n_out are attributes on the underlying ensemble
    assert mb.model.n_input == N_FEATURES
    assert mb.model.n_out == 1


def test_model_is_none_before_train():
    """model attribute is None until train() is called."""
    mb = TorchMLPModelBox(ensemble_size=N_ENSEMBLE)
    assert mb.model is None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_weights_change_after_training(train_val_split, tmp_path):
    """Gradient descent actually updates weights — model is not frozen."""
    train_df, val_df = train_val_split
    mb = TorchMLPModelBox(ensemble_size=N_ENSEMBLE, n_units=16, n_hidden=2)
    mb.load_training_data(
        features=FEATURES, labels=LABEL, data=train_df,
        cache=str(tmp_path / "cache"),
    )
    mb.input_shape = (N_FEATURES,)
    mb.output_shape = (1,)
    mb.model = mb.create_model()
    # Snapshot weights BEFORE training
    before = {k: v.clone() for k, v in mb.model.state_dict().items()}
    mb.train(val_data=val_df, epochs=2, batch_size=16, cache=str(tmp_path / "cache"))
    after = mb.model.state_dict()
    # At least one parameter tensor must differ
    any_changed = any(
        not before[k].equal(after[k]) for k in before
    )
    assert any_changed, "No parameters changed during training — optimiser may be broken"


def test_load_training_data_sets_shapes(train_val_split, tmp_path):
    """load_training_data() correctly infers input and output shapes."""
    train_df, _ = train_val_split
    mb = TorchMLPModelBox(ensemble_size=N_ENSEMBLE)
    mb.load_training_data(
        features=FEATURES, labels=LABEL, data=train_df,
        cache=str(tmp_path / "cache"),
    )
    assert mb.input_shape == (N_FEATURES,)
    assert mb.output_shape == (1,)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_predict_output_has_prediction_column(trained_mlp, train_val_split):
    """predict() returns a Dataset containing the 'prediction' column."""
    _, val_df = train_val_split
    result = trained_mlp.predict(data=val_df)
    assert "prediction" in result.column_names


def test_predict_ensemble_raw_shape(trained_mlp, train_val_split):
    """Without aggregator, prediction shape is (n_samples, ensemble_size)."""
    _, val_df = train_val_split
    result = trained_mlp.predict(data=val_df)
    result = result.with_format("numpy")
    preds = result["prediction"]
    assert preds.shape == (N_VAL, N_ENSEMBLE), (
        f"Expected ({N_VAL}, {N_ENSEMBLE}), got {preds.shape}"
    )


def test_predict_output_shape_with_aggregator(trained_mlp, train_val_split):
    """With aggregator='mean', prediction is squeezed to (n_samples,)."""
    _, val_df = train_val_split
    result = trained_mlp.predict(data=val_df, aggregator="mean")
    result = result.with_format("numpy")
    preds = result["prediction"]
    assert preds.shape == (N_VAL,), (
        f"Expected ({N_VAL},), got {preds.shape}"
    )


def test_predict_preserves_input_columns(trained_mlp, train_val_split):
    """All original DataFrame columns survive into the output Dataset."""
    _, val_df = train_val_split
    result = trained_mlp.predict(data=val_df)
    for col in FEATURES + [LABEL]:
        assert col in result.column_names, f"Column '{col}' missing from prediction output"


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_checkpoint_files_created(trained_mlp, tmp_path):
    """save_checkpoint() writes the expected set of files."""
    ckpt = str(tmp_path / "ckpt")
    trained_mlp.save_checkpoint(ckpt)
    files = set(os.listdir(ckpt))
    expected = {
        "modelbox-init-config.json",
        "modelbox-special-args.json",
        "model-config.json",
        "data-config.json",
        "params.pt",
    }
    assert expected.issubset(files), (
        f"Missing checkpoint files: {expected - files}"
    )


def test_checkpoint_model_config_roundtrips_ensemble_size(trained_mlp, tmp_path):
    """model-config.json preserves the ensemble_size kwarg."""
    ckpt = str(tmp_path / "ckpt_cfg")
    trained_mlp.save_checkpoint(ckpt)
    with open(os.path.join(ckpt, "model-config.json")) as f:
        cfg = json.load(f)
    assert "ensemble_size" in cfg
    assert cfg["ensemble_size"] == N_ENSEMBLE


def test_checkpoint_roundtrip_predictions(trained_mlp, train_val_split, tmp_path):
    """Predictions before save are byte-for-byte identical after load."""
    _, val_df = train_val_split
    ckpt = str(tmp_path / "ckpt_rt")
    trained_mlp.save_checkpoint(ckpt)

    # Predictions before reload
    preds_before = (
        trained_mlp.predict(data=val_df)
        .with_format("numpy")["prediction"]
    )

    # Load into a fresh instance
    mb2 = TorchMLPModelBox(ensemble_size=N_ENSEMBLE)
    mb2.load_checkpoint(ckpt)
    preds_after = (
        mb2.predict(data=val_df)
        .with_format("numpy")["prediction"]
    )

    np.testing.assert_array_equal(
        preds_before, preds_after,
        err_msg="Predictions changed after checkpoint roundtrip",
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def test_evaluate_returns_expected_metric_keys(trained_mlp, train_val_split):
    """evaluate() returns a dict with rmse, pearson_r, spearman_rho keys."""
    _, val_df = train_val_split
    _, metrics = trained_mlp.evaluate(data=val_df, aggregator="mean")
    for key in ("rmse", "pearson_r", "spearman_rho"):
        assert key in metrics, f"Metric '{key}' missing from evaluate() output"


def test_evaluate_perfect_predictions_zero_rmse(train_val_split, tmp_path):
    """When predictions equal labels exactly, RMSE must be 0.

    We test this by monkey-patching the model to return the true labels,
    bypassing training entirely — this avoids any tautology about
    the training process.
    """
    import torch
    train_df, val_df = train_val_split

    mb = TorchMLPModelBox(ensemble_size=1, n_units=8, n_hidden=1)
    mb.load_training_data(
        features=FEATURES, labels=LABEL, data=train_df,
        cache=str(tmp_path / "cache_perf"),
    )
    mb.train(
        val_data=val_df, epochs=1, batch_size=16,
        cache=str(tmp_path / "cache_perf"),
    )

    # Patch the forward pass to return exact labels as predictions
    true_labels = torch.tensor(val_df[LABEL].values, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)

    class _PerfectModel:
        def __call__(self, x):
            return true_labels
        def eval(self): pass
        def train(self): pass

    mb.model = _PerfectModel()
    _, metrics = mb.evaluate(data=val_df, aggregator="mean")
    assert metrics["rmse"] == pytest.approx(0.0, abs=1e-5)
