"""Shared pytest fixtures for duvidnn tests."""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Synthetic tabular data (no chemistry deps)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def linear_df(rng):
    """50-row DataFrame: 4 features, 1 label = known linear combination + noise.

    label = 2*f1 - f2 + 0.5*f3 + noise(scale=0.05)
    The ground-truth is known, so we can write non-trivial assertions about
    whether a trained model has learned anything useful.
    """
    n = 60
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    f4 = rng.normal(size=n)
    y = 2.0 * f1 - f2 + 0.5 * f3 + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "f4": f4, "y": y})


@pytest.fixture(scope="session")
def train_val_split(linear_df):
    return linear_df.iloc[:50].copy(), linear_df.iloc[50:].copy()


@pytest.fixture
def trained_mlp(train_val_split, tmp_path):
    """TorchMLPModelBox trained for 2 epochs on the linear synthetic dataset."""
    from duvidnn.torch.modelbox.modelboxes import TorchMLPModelBox
    train_df, val_df = train_val_split
    mb = TorchMLPModelBox(ensemble_size=2, n_units=16, n_hidden=2)
    mb.load_training_data(
        features=["f1", "f2", "f3", "f4"],
        labels="y",
        data=train_df,
        cache=str(tmp_path / "cache"),
    )
    mb.train(
        val_data=val_df,
        epochs=2,
        batch_size=16,
        cache=str(tmp_path / "cache"),
    )
    return mb


# ---------------------------------------------------------------------------
# SMILES data for Chemprop tests (no-op if chemprop not installed)
# ---------------------------------------------------------------------------

SMILES_LIST = [
    "CC(=O)Nc1ccc(O)cc1",          # paracetamol
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "c1ccc2c(c1)ccc3cccc4ccc5ccccc5c1ccc2c3c41",  # pyrene
    "CC1=CC2=C(C=C1)N=CC2=O",
    "O=C(O)c1ccccc1O",             # salicylic acid
    "c1ccc(cc1)N",                  # aniline
    "CCO",                          # ethanol
    "c1ccccc1",                     # benzene
    "CC(=O)O",                      # acetic acid
]
# Synthetic IC50 values (μM), loosely correlated with MW
IC50_VALUES = [151.2, 0.8, 5.4, 2200.0, 43.0, 89.0, 310.0, 9999.0, 120.0, 1500.0]


@pytest.fixture(scope="session")
def smiles_df():
    """10-molecule DataFrame with SMILES and synthetic IC50 values."""
    pytest.importorskip("chemprop", reason="chemprop not installed")
    return pd.DataFrame({
        "smiles": SMILES_LIST,
        "ic50": IC50_VALUES,
    })
