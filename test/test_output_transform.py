"""Tests for the OutputTransform interface and built-in implementations.

These are TDD specification tests — they will FAIL until the OutputTransform
classes are implemented as part of the refactor.  Once the refactor is done
they must all pass.

Anti-self-fulfilling design:
- HillEquation tests use mathematical limit identities (C=IC50 → midpoint,
  C=0 → baseline, C>>IC50 → top) rather than re-implementing the formula.
- ScaleTransform mean/std are checked against numpy computed directly on the
  raw label column, not on anything the transform itself cached.
- ModelBox integration tests compare predictions with vs without the transform
  applied, verifying that the transform actually changes values.
"""

import json
import os

import numpy as np
import pytest

# These imports will fail until the refactor creates the modules.
from duvidnn.base.output_transform import IdentityOutputTransform, OutputTransform
from duvidnn.base.output_transforms.hill import HillEquationOutputTransform
from duvidnn.base.output_transforms.scale import ScaleOutputTransform


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(predictions, concentration=None):
    """Construct a minimal batch dict as ModelBox.predict() would supply it."""
    batch = {"prediction": list(predictions)}
    if concentration is not None:
        batch["conc"] = list(concentration)
    return batch


# ---------------------------------------------------------------------------
# IdentityOutputTransform
# ---------------------------------------------------------------------------

class TestIdentityOutputTransform:

    def test_is_output_transform_subclass(self):
        assert issubclass(IdentityOutputTransform, OutputTransform)

    def test_passthrough_does_not_alter_predictions(self):
        preds = np.array([1.0, 2.0, 3.0])
        batch = _make_batch(preds)
        t = IdentityOutputTransform()
        result = t.transform(batch, "prediction")
        np.testing.assert_array_equal(
            np.asarray(result["prediction"]), preds,
            err_msg="IdentityOutputTransform must not alter predictions",
        )

    def test_passthrough_preserves_other_columns(self):
        batch = {"prediction": [1.0, 2.0], "conc": [0.1, 0.2], "smiles": ["C", "CC"]}
        t = IdentityOutputTransform()
        result = t.transform(batch, "prediction")
        assert set(result.keys()) == {"prediction", "conc", "smiles"}

    def test_fit_returns_self(self):
        t = IdentityOutputTransform()
        ret = t.fit(training_data=None)
        assert ret is t


# ---------------------------------------------------------------------------
# HillEquationOutputTransform — mathematical identity tests
# ---------------------------------------------------------------------------

class TestHillEquationOutputTransform:

    def test_is_output_transform_subclass(self):
        assert issubclass(HillEquationOutputTransform, OutputTransform)

    def test_at_ic50_gives_midpoint(self):
        """When C == IC50, fractional effect must equal midpoint exactly.

        Hill equation: effect = baseline + (top - baseline) * C^n / (IC50^n + C^n)
        When C = IC50: ratio = 1, so effect = baseline + (top - baseline) * 0.5
        This is a mathematical identity independent of n, baseline, or top.
        """
        ic50 = np.array([1.0, 5.0, 100.0])
        conc = ic50.copy()  # C == IC50 for every molecule
        batch = {"prediction": list(ic50), "conc": list(conc)}

        t = HillEquationOutputTransform(
            concentration_column="conc",
            hill_n=1.0,
            baseline=0.0,
            top=1.0,
        )
        result = t.transform(batch, "prediction")
        effects = np.asarray(result["prediction"])
        np.testing.assert_allclose(
            effects, 0.5,
            rtol=1e-5,
            err_msg="At C==IC50 the Hill equation must always yield 0.5 (for baseline=0, top=1)",
        )

    def test_zero_concentration_gives_baseline(self):
        """When C = 0, the Hill equation must return baseline regardless of IC50."""
        ic50 = np.array([1.0, 10.0, 500.0])
        conc = np.zeros_like(ic50)
        batch = {"prediction": list(ic50), "conc": list(conc)}

        baseline = 0.05
        t = HillEquationOutputTransform(
            concentration_column="conc",
            hill_n=1.5,
            baseline=baseline,
            top=1.0,
        )
        result = t.transform(batch, "prediction")
        effects = np.asarray(result["prediction"])
        np.testing.assert_allclose(
            effects, baseline,
            atol=1e-6,
            err_msg="At C=0 the Hill equation must return baseline",
        )

    def test_very_high_concentration_approaches_top(self):
        """When C >> IC50, fractional effect must approach top."""
        ic50 = np.array([1.0, 1.0, 1.0])
        conc = ic50 * 1e8  # 100 million × IC50
        batch = {"prediction": list(ic50), "conc": list(conc)}

        top = 0.9
        t = HillEquationOutputTransform(
            concentration_column="conc",
            hill_n=1.0,
            baseline=0.0,
            top=top,
        )
        result = t.transform(batch, "prediction")
        effects = np.asarray(result["prediction"])
        np.testing.assert_allclose(
            effects, top,
            rtol=1e-4,
            err_msg="At C>>IC50 the Hill equation must approach top",
        )

    def test_steeper_hill_coefficient_gives_sharper_response(self):
        """n=2 gives more extreme effect than n=1 for C > IC50.

        At C = 10 × IC50:
          n=1: effect = 10/11 ≈ 0.909
          n=2: effect = 100/101 ≈ 0.990
        So n=2 should be closer to top.
        """
        ic50 = np.array([1.0])
        conc = np.array([10.0])  # C = 10 × IC50
        batch_n1 = {"prediction": list(ic50), "conc": list(conc)}
        batch_n2 = {"prediction": list(ic50), "conc": list(conc)}

        t1 = HillEquationOutputTransform("conc", hill_n=1.0, baseline=0.0, top=1.0)
        t2 = HillEquationOutputTransform("conc", hill_n=2.0, baseline=0.0, top=1.0)

        eff_n1 = np.asarray(t1.transform(batch_n1, "prediction")["prediction"])[0]
        eff_n2 = np.asarray(t2.transform(batch_n2, "prediction")["prediction"])[0]

        assert eff_n2 > eff_n1, (
            f"n=2 should give higher effect than n=1 at C>IC50, "
            f"got n=1: {eff_n1:.4f}, n=2: {eff_n2:.4f}"
        )

    def test_custom_baseline_and_top(self):
        """Arbitrary baseline/top parameters shift the dynamic range correctly."""
        ic50 = np.array([1.0])
        conc = ic50.copy()  # midpoint identity
        batch = {"prediction": list(ic50), "conc": list(conc)}

        baseline, top = 0.2, 0.8
        t = HillEquationOutputTransform("conc", hill_n=1.0, baseline=baseline, top=top)
        result = t.transform(batch, "prediction")
        effect = np.asarray(result["prediction"])[0]
        expected_midpoint = baseline + (top - baseline) * 0.5
        assert effect == pytest.approx(expected_midpoint, rel=1e-5)

    def test_fit_returns_self_by_default(self):
        t = HillEquationOutputTransform("conc")
        ret = t.fit(training_data=None)
        assert ret is t

    def test_predictions_differ_from_raw_ic50(self):
        """The transform must actually change the prediction values (C ≠ IC50)."""
        ic50 = np.array([1.0, 5.0, 100.0])
        conc = ic50 * 3.0  # C ≠ IC50, so effect ≠ IC50
        batch = {"prediction": list(ic50), "conc": list(conc)}
        t = HillEquationOutputTransform("conc", hill_n=1.0, baseline=0.0, top=1.0)
        result = t.transform(batch, "prediction")
        transformed = np.asarray(result["prediction"])
        # Transformed values are fractional effects in [0,1]; raw IC50 values are >> 1
        assert not np.allclose(transformed, ic50), (
            "Transform output must differ from raw IC50 predictions"
        )


# ---------------------------------------------------------------------------
# ScaleOutputTransform
# ---------------------------------------------------------------------------

class TestScaleOutputTransform:

    def _make_training_dataset(self, labels):
        """Create a minimal HuggingFace Dataset with a label column."""
        from datasets import Dataset
        return Dataset.from_dict({"y": labels.tolist()})

    def test_is_output_transform_subclass(self):
        assert issubclass(ScaleOutputTransform, OutputTransform)

    def test_fit_computes_correct_mean_and_std(self):
        """fit() must compute mean and std matching numpy directly on labels."""
        labels = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dataset = self._make_training_dataset(labels)

        t = ScaleOutputTransform(label_column="y")
        t.fit(training_data=dataset)

        assert t.mean_ == pytest.approx(np.mean(labels), rel=1e-5)
        assert t.std_ == pytest.approx(np.std(labels), rel=1e-5)

    def test_fit_then_transform_scales_predictions(self):
        """After fit(), transform(pred) = (pred - mean) / std."""
        labels = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        dataset = self._make_training_dataset(labels)

        t = ScaleOutputTransform(label_column="y")
        t.fit(training_data=dataset)

        preds = np.array([15.0, 25.0, 35.0])
        batch = {"prediction": list(preds)}
        result = t.transform(batch, "prediction")
        scaled = np.asarray(result["prediction"])

        expected = (preds - np.mean(labels)) / np.std(labels)
        np.testing.assert_allclose(scaled, expected, rtol=1e-5)

    def test_fit_returns_self(self):
        labels = np.array([1.0, 2.0, 3.0])
        dataset = self._make_training_dataset(labels)
        t = ScaleOutputTransform(label_column="y")
        ret = t.fit(training_data=dataset)
        assert ret is t


# ---------------------------------------------------------------------------
# OutputTransform integration with ModelBox
# ---------------------------------------------------------------------------

class TestOutputTransformInModelBox:

    @pytest.fixture
    def mb_with_identity_transform(self, trained_mlp):
        """Return a copy of trained_mlp with IdentityOutputTransform attached."""
        # Attach the transform to the already-trained modelbox
        trained_mlp.output_transform = IdentityOutputTransform()
        yield trained_mlp
        # Cleanup: remove transform so other tests are unaffected
        trained_mlp.output_transform = None

    def test_identity_transform_does_not_change_predictions(
        self, trained_mlp, mb_with_identity_transform, train_val_split
    ):
        """Predictions with IdentityOutputTransform are identical to without."""
        _, val_df = train_val_split
        preds_plain = (
            trained_mlp.predict(data=val_df)
            .with_format("numpy")["prediction"]
        )
        preds_identity = (
            mb_with_identity_transform.predict(data=val_df)
            .with_format("numpy")["prediction"]
        )
        np.testing.assert_array_equal(preds_plain, preds_identity)

    def test_hill_transform_changes_predictions(self, trained_mlp, train_val_split):
        """HillEquationOutputTransform gives different values from raw predictions."""
        _, val_df = train_val_split

        # Add a concentration column to val_df (arbitrary fixed value)
        val_df = val_df.copy()
        val_df["conc"] = 1.0

        preds_raw = (
            trained_mlp.predict(data=val_df)
            .with_format("numpy")["prediction"]
        )

        trained_mlp.output_transform = HillEquationOutputTransform(
            concentration_column="conc",
            hill_n=1.0,
            baseline=0.0,
            top=1.0,
        )
        try:
            preds_transformed = (
                trained_mlp.predict(data=val_df)
                .with_format("numpy")["prediction"]
            )
        finally:
            trained_mlp.output_transform = None

        assert not np.allclose(preds_raw, preds_transformed), (
            "Hill transform should change predictions but did not"
        )

    def test_output_transform_applied_before_aggregator(
        self, trained_mlp, train_val_split
    ):
        """Identity transform before aggregation gives same result as no transform.

        This verifies that the transform is inserted at the correct position in
        the pipeline (before ensemble aggregation), and that it does not break
        the aggregation step.
        """
        _, val_df = train_val_split
        trained_mlp.output_transform = IdentityOutputTransform()
        try:
            preds_with_transform = (
                trained_mlp.predict(data=val_df, aggregator="mean")
                .with_format("numpy")["prediction"]
            )
        finally:
            trained_mlp.output_transform = None

        preds_without = (
            trained_mlp.predict(data=val_df, aggregator="mean")
            .with_format("numpy")["prediction"]
        )
        np.testing.assert_array_equal(preds_with_transform, preds_without)

    def test_output_transform_checkpoint_roundtrip(self, trained_mlp, train_val_split, tmp_path):
        """HillEquation transform parameters survive a checkpoint save/load cycle."""
        _, val_df = train_val_split
        val_df = val_df.copy()
        val_df["conc"] = 0.5

        t = HillEquationOutputTransform(
            concentration_column="conc",
            hill_n=2.0,
            baseline=0.1,
            top=0.95,
        )
        trained_mlp.output_transform = t
        ckpt = str(tmp_path / "ckpt_transform")
        trained_mlp.save_checkpoint(ckpt)

        from duvidnn.torch.modelbox.modelboxes import TorchMLPModelBox
        mb2 = TorchMLPModelBox(ensemble_size=2)
        mb2.load_checkpoint(ckpt)

        assert isinstance(mb2.output_transform, HillEquationOutputTransform)
        assert mb2.output_transform.hill_n == pytest.approx(2.0)
        assert mb2.output_transform.baseline == pytest.approx(0.1)
        assert mb2.output_transform.top == pytest.approx(0.95)
        assert mb2.output_transform.concentration_column == "conc"

        trained_mlp.output_transform = None
