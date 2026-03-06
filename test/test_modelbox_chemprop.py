"""Integration tests for ChempropModelBox.

All tests are skipped automatically when chemprop is not installed.
They must pass on the current (pre-refactor) code and continue to pass
after the refactor.
"""

import numpy as np
import pytest

chemprop = pytest.importorskip("chemprop", reason="chemprop not installed")

from duvidnn.torch.modelbox.modelboxes import ChempropModelBox

SMILES_COL = "smiles"
LABEL_COL = "ic50"


@pytest.fixture
def trained_chemprop(smiles_df, tmp_path):
    """ChempropModelBox trained for 1 epoch on the 10-molecule SMILES fixture."""
    train_df = smiles_df.iloc[:8].copy()
    val_df = smiles_df.iloc[8:].copy()
    mb = ChempropModelBox(ensemble_size=1)
    mb.load_training_data(
        structure_column=SMILES_COL,
        labels=LABEL_COL,
        data=train_df,
        cache=str(tmp_path / "cache"),
    )
    mb.train(
        val_data=val_df,
        epochs=1,
        batch_size=4,
        cache=str(tmp_path / "cache"),
    )
    return mb


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def test_chemprop_special_args_set(smiles_df, tmp_path):
    """_special_args['chemprop_input_column'] is a non-empty string after load."""
    mb = ChempropModelBox(ensemble_size=1)
    mb.load_training_data(
        structure_column=SMILES_COL,
        labels=LABEL_COL,
        data=smiles_df,
        cache=str(tmp_path / "cache_sa"),
    )
    col = mb._special_args.get("chemprop_input_column")
    assert isinstance(col, str) and len(col) > 0, (
        f"Expected non-empty chemprop_input_column, got {col!r}"
    )


def test_chemprop_training_example_has_graph_input_column(smiles_df, tmp_path):
    """After load_training_data(), training_example contains the model input key."""
    mb = ChempropModelBox(ensemble_size=1)
    mb.load_training_data(
        structure_column=SMILES_COL,
        labels=LABEL_COL,
        data=smiles_df,
        cache=str(tmp_path / "cache_te"),
    )
    # The collated training data must contain a column matching _in_key
    in_key = mb._in_key
    matching = [c for c in mb.training_example.column_names if c.startswith(in_key)]
    assert len(matching) >= 1, (
        f"No column starting with '{in_key}' found in training_example columns: "
        f"{mb.training_example.column_names}"
    )


def test_chemprop_no_extra_featurizers_gives_zero_input_shape(smiles_df, tmp_path):
    """With no extra featurizers, input_shape should be (0,) — no x_d."""
    mb = ChempropModelBox(ensemble_size=1)
    mb.load_training_data(
        structure_column=SMILES_COL,
        labels=LABEL_COL,
        data=smiles_df,
        cache=str(tmp_path / "cache_0"),
    )
    # input_shape is set inside create_model for chemprop, so call it
    mb.model = mb.create_model()
    assert mb.input_shape == (0,), (
        f"Expected input_shape (0,) with no extra featurizers, got {mb.input_shape}"
    )


def test_chemprop_with_fp_gives_nonzero_input_shape(smiles_df, tmp_path):
    """use_fp=True adds Morgan fingerprint as x_d, so input_shape > (0,)."""
    mb = ChempropModelBox(ensemble_size=1, use_fp=True)
    mb.load_training_data(
        structure_column=SMILES_COL,
        labels=LABEL_COL,
        data=smiles_df,
        cache=str(tmp_path / "cache_fp"),
    )
    mb.model = mb.create_model()
    assert mb.input_shape[0] > 0, (
        f"Expected positive input_shape with use_fp=True, got {mb.input_shape}"
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def test_chemprop_predict_returns_prediction_column(trained_chemprop, smiles_df):
    """predict() completes and adds a 'prediction' column."""
    result = trained_chemprop.predict(data=smiles_df)
    assert "prediction" in result.column_names


def test_chemprop_predict_output_shape(trained_chemprop, smiles_df):
    """Raw predictions have shape (n_molecules, ensemble_size)."""
    n = len(smiles_df)
    ensemble_size = 1
    result = trained_chemprop.predict(data=smiles_df).with_format("numpy")
    preds = result["prediction"]
    assert preds.shape == (n, ensemble_size), (
        f"Expected ({n}, {ensemble_size}), got {preds.shape}"
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def test_chemprop_checkpoint_roundtrip_predictions(trained_chemprop, smiles_df, tmp_path):
    """Predictions before save are identical to predictions after load."""
    ckpt = str(tmp_path / "chemprop_ckpt")
    trained_chemprop.save_checkpoint(ckpt)

    preds_before = (
        trained_chemprop.predict(data=smiles_df)
        .with_format("numpy")["prediction"]
    )

    mb2 = ChempropModelBox(ensemble_size=1)
    mb2.load_checkpoint(ckpt)
    preds_after = (
        mb2.predict(data=smiles_df)
        .with_format("numpy")["prediction"]
    )

    np.testing.assert_array_equal(
        preds_before, preds_after,
        err_msg="Chemprop predictions changed after checkpoint roundtrip",
    )


def test_chemprop_checkpoint_restores_special_args(trained_chemprop, smiles_df, tmp_path):
    """chemprop_input_column survives a checkpoint roundtrip."""
    ckpt = str(tmp_path / "chemprop_ckpt_sa")
    original_col = trained_chemprop._special_args["chemprop_input_column"]
    trained_chemprop.save_checkpoint(ckpt)

    mb2 = ChempropModelBox(ensemble_size=1)
    mb2.load_checkpoint(ckpt)
    assert mb2._special_args["chemprop_input_column"] == original_col
