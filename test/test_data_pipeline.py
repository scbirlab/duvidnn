"""Tests for the DataPipeline.get_model_input_info() interface.

These are TDD specification tests — they will FAIL until the DataPipeline
refactor creates the required interfaces. Once the refactor is done they
must all pass.

The tests verify:
1. TorchDataPipeline.get_model_input_info() returns a dict describing the
   model's required input dimensions after training data has been loaded.
2. ChempropDataPipeline.get_model_input_info() returns (n_input=0, n_out=1)
   when there are no extra featurizers (no x_d), and n_input>0 when extras
   are present.
3. ChempropDataPipeline.graph_input_column is a first-class string attribute
   (not buried in _special_args) and matches the old _special_args value.
"""

import numpy as np
import pandas as pd
import pytest

# These imports will fail until the refactor creates the modules.
from duvidnn.torch.modelbox.data import TorchDataPipeline, ChempropDataPipeline

FEATURES = ["f1", "f2", "f3", "f4"]
LABEL = "y"
N_FEATURES = 4

SMILES_COL = "smiles"
LABEL_COL = "ic50"
SMILES_LIST = [
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "O=C(O)c1ccccc1O",
    "c1ccc(cc1)N",
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
]
IC50_VALUES = [151.2, 0.8, 5.4, 89.0, 310.0, 9999.0, 120.0, 1500.0]


@pytest.fixture
def tabular_df(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    data = {f: rng.normal(size=n) for f in FEATURES}
    data[LABEL] = 2.0 * data["f1"] - data["f2"]
    return pd.DataFrame(data)


@pytest.fixture
def smiles_frame():
    pytest.importorskip("chemprop", reason="chemprop not installed")
    return pd.DataFrame({"smiles": SMILES_LIST, "ic50": IC50_VALUES})


# ---------------------------------------------------------------------------
# TorchDataPipeline
# ---------------------------------------------------------------------------

class TestTorchDataPipeline:

    def test_get_model_input_info_returns_dict(self, tabular_df, tmp_path):
        """get_model_input_info() returns a dict after data is loaded."""
        pipeline = TorchDataPipeline()
        pipeline.load_training_data(
            features=FEATURES, labels=LABEL, data=tabular_df,
            cache=str(tmp_path / "cache"),
        )
        info = pipeline.get_model_input_info()
        assert isinstance(info, dict), (
            f"get_model_input_info() must return dict, got {type(info)}"
        )

    def test_get_model_input_info_n_input(self, tabular_df, tmp_path):
        """n_input matches the number of feature columns."""
        pipeline = TorchDataPipeline()
        pipeline.load_training_data(
            features=FEATURES, labels=LABEL, data=tabular_df,
            cache=str(tmp_path / "cache"),
        )
        info = pipeline.get_model_input_info()
        assert info["n_input"] == N_FEATURES, (
            f"Expected n_input={N_FEATURES}, got {info['n_input']}"
        )

    def test_get_model_input_info_n_out(self, tabular_df, tmp_path):
        """n_out matches the number of label columns (1 here)."""
        pipeline = TorchDataPipeline()
        pipeline.load_training_data(
            features=FEATURES, labels=LABEL, data=tabular_df,
            cache=str(tmp_path / "cache"),
        )
        info = pipeline.get_model_input_info()
        assert info["n_out"] == 1, (
            f"Expected n_out=1, got {info['n_out']}"
        )


# ---------------------------------------------------------------------------
# ChempropDataPipeline
# ---------------------------------------------------------------------------

class TestChempropDataPipeline:

    def test_get_model_input_info_no_extra_featurizers(self, smiles_frame, tmp_path):
        """Without extra featurizers, n_input == 0 (no x_d)."""
        pipeline = ChempropDataPipeline()
        pipeline.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_0"),
        )
        info = pipeline.get_model_input_info()
        assert info["n_input"] == 0, (
            f"Expected n_input=0 with no extra featurizers, got {info['n_input']}"
        )

    def test_get_model_input_info_with_fp_gives_nonzero_n_input(self, smiles_frame, tmp_path):
        """With use_fp=True, n_input > 0 (Morgan fingerprint in x_d)."""
        pipeline = ChempropDataPipeline(use_fp=True)
        pipeline.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_fp"),
        )
        info = pipeline.get_model_input_info()
        assert info["n_input"] > 0, (
            f"Expected n_input>0 with use_fp=True, got {info['n_input']}"
        )

    def test_get_model_input_info_n_out(self, smiles_frame, tmp_path):
        """n_out == 1 for a single label column."""
        pipeline = ChempropDataPipeline()
        pipeline.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_out"),
        )
        info = pipeline.get_model_input_info()
        assert info["n_out"] == 1

    def test_graph_input_column_is_string_attribute(self, smiles_frame, tmp_path):
        """graph_input_column is a first-class string attribute on the pipeline."""
        pipeline = ChempropDataPipeline()
        pipeline.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_col"),
        )
        col = pipeline.graph_input_column
        assert isinstance(col, str) and len(col) > 0, (
            f"graph_input_column must be a non-empty string, got {col!r}"
        )

    def test_graph_input_column_matches_old_special_args(self, smiles_frame, tmp_path):
        """graph_input_column equals what _special_args['chemprop_input_column']
        used to contain — ensuring backward-compatible semantics after refactor.

        We derive the expected value from a ChempropModelBox (old API) and
        compare it against the new pipeline attribute.
        """
        from duvidnn.torch.modelbox.modelboxes import ChempropModelBox

        # Old API: derive the column name
        mb = ChempropModelBox(ensemble_size=1)
        mb.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_old"),
        )
        old_col = mb._special_args["chemprop_input_column"]

        # New API: pipeline attribute
        pipeline = ChempropDataPipeline()
        pipeline.load_training_data(
            structure_column=SMILES_COL,
            labels=LABEL_COL,
            data=smiles_frame,
            cache=str(tmp_path / "cache_new"),
        )
        assert pipeline.graph_input_column == old_col, (
            f"graph_input_column ({pipeline.graph_input_column!r}) does not match "
            f"old _special_args value ({old_col!r})"
        )
