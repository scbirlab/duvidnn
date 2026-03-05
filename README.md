# duvidnn

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/duvidnn/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/duvidnn)
![PyPI](https://img.shields.io/pypi/v/duvidnn)

**duvidnn** is a Python toolkit for calculating confidence and information metrics
for deep learning models. It wraps general-purpose, chemistry-specific, and taxonomic
neural networks in a `ModelBox` abstraction that keeps every model co-located with
its training data, so that uncertainty measures that depend on what the model has
already seen can be computed in one place.

As a bonus, **duvidnn** provides a command-line interface for training, predicting,
data splitting, and hyperparameter search.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Command-line interface](#command-line-interface)
- [Python API](#python-api)
    - [ModelBox overview](#modelbox-overview)
    - [Training](#training)
    - [Saving and loading checkpoints](#saving-and-loading-checkpoints)
    - [Prediction and evaluation](#prediction-and-evaluation)
    - [Uncertainty and information metrics](#uncertainty-and-information-metrics)
- [Architecture](#architecture)
    - [Package layout](#package-layout)
    - [Class hierarchy](#class-hierarchy)
    - [Preprocessing pipeline](#preprocessing-pipeline)
    - [Information metric internals](#information-metric-internals)
- [Extending duvidnn](#extending-duvidnn)
    - [Adding a new ModelBox](#adding-a-new-modelbox)
    - [Adding a new preprocessing function](#adding-a-new-preprocessing-function)
- [Known issues](#known-issues)
- [Issues, problems, suggestions](#issues-problems-suggestions)

## Installation

### From PyPI

```bash
pip install duvidnn
```

Optional extras for chemistry ML:

```bash
pip install duvidnn[chem]
```

For taxonomic embeddings via [vectome](https://github.com/scbirlab/vectome):

```bash
pip install duvidnn[bio]
```

For dataset splitting utilities (scaffold, FAISS):

```bash
pip install duvidnn[splits]
```

You can combine extras:

```bash
pip install duvidnn[bio,chem,splits]
```

### From source

```bash
git clone https://github.com/scbirlab/duvidnn.git
cd duvidnn
pip install -e ".[dev]"
```

## Quick start

```python
from duvidnn.autoclass import AutoModelBox

# Create a fingerprint model
modelbox = AutoModelBox("fingerprint", n_units=16, n_hidden=2, ensemble_size=10)._instance

# Load data (HuggingFace dataset, local CSV, or pandas DataFrame)
modelbox.load_training_data(
    data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train",
    structure_column="smiles",
    labels="clogp",
)

# Train
modelbox.train(
    val_data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    epochs=10,
    batch_size=128,
)

# Predict with uncertainty
predictions = modelbox.predict(
    data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
)
doubtscore = modelbox.doubtscore(
    candidates="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
)
```

## Command-line interface

**duvidnn** provides five subcommands:

```
duvidnn [-h] [--version] {hyperprep,train,predict,split,percentiles} ...
```

Get help for any subcommand with `duvidnn <command> --help`.

### Model training

```bash
duvidnn train \
    -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --model-class fingerprint \
    --structure smiles \
    --labels clogp \
    --ensemble-size 10 \
    --epochs 10 \
    --learning-rate 0.001 \
    --output model.dv
```

Available model classes: `mlp`, `fingerprint`, `bilinear`, `bilinear-fp`, `chemprop`, `cnn`.

### Prediction with uncertainty

```bash
duvidnn predict \
    --test hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --checkpoint model.dv \
    --labels clogp \
    --variance \
    --doubtscore \
    --output predictions.parquet
```

Uncertainty options:

| Flag | Description |
|------|-------------|
| `--variance` | Ensemble prediction variance |
| `--tanimoto` | Tanimoto nearest-neighbor distance to training set (chemistry models) |
| `--doubtscore` | Doubtscore |
| `--information-sensitivity` | Information sensitivity |
| `--last-layer` | Restrict gradient computation to the output layer (large speed-up) |
| `--optimality` | Assume model is trained to gradient zero (faster information sensitivity) |
| `--approx` | Hessian approximation method: `bekas`, `exact_diagonal`, `squared_jacobian`, `rough_finite_difference` |

Output format is inferred from the `--output` file extension (CSV, Parquet, Arrow, HF Dataset).

Use `--start` and `--end` to restrict prediction to a row range, which is useful for
parallelizing across chunks.

### Data splitting

Out-of-memory scaffold and approximate spectral (FAISS) splitting:

```bash
duvidnn split \
    hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --train .7 --validation .15 \
    --structure smiles \
    --type faiss \
    --seed 1 \
    --output faiss.csv \
    --plot faiss.png
```

### Percentile annotation

Tag rows that fall in the top percentiles of specified columns. Works on datasets
that do not fit in memory.

```bash
duvidnn percentiles \
    hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --columns clogp tpsa \
    --percentiles 1 5 10 \
    --output percentiles.parquet \
    --plot percentiles-plot.png \
    --structure smiles
```

### Hyperparameter preparation

Generate all combinations of hyperparameters, then index into them for systematic
or parallel search:

```bash
printf '{"model_class": "fingerprint", "use_2d": [true, false], "n_units": 16}' \
    | duvidnn hyperprep -o hyperopt.json

duvidnn train \
    -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    -c hyperopt.json -i 0 \
    --output model.dv
```

### Input data formats

In all CLI commands, input data can be:
- A local file in CSV, Parquet, Arrow, or HF Dataset format
- A remote HuggingFace dataset, indicated by the `hf://` prefix (e.g.
  `hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train`, where `@` selects
  the configuration and `:` selects the split)

## Python API

### ModelBox overview

The central abstraction is the `ModelBox`: a container that keeps a model together
with its training data. This coupling is fundamental because uncertainty metrics
like doubtscore and information sensitivity depend on the training data distribution.

```python
from duvidnn.autoclass import AutoModelBox, MODELBOX_REGISTRY
from pprint import pprint

pprint(MODELBOX_REGISTRY)
# {'bilinear': TorchBilinearModelBox,
#  'bilinear-fp': TorchBilinearFingerprintModelBox,
#  'chemprop': ChempropModelBox,
#  'cnn': TorchCNN2DModelBox,
#  'fingerprint': TorchFingerprintModelBox,
#  'mlp': TorchMLPModelBox}
```

| ModelBox | Input | Description |
|----------|-------|-------------|
| `mlp` | Numeric features | General-purpose multilayer perceptron ensemble |
| `fingerprint` | SMILES | MLP on Morgan fingerprints and/or 2D descriptors |
| `bilinear` | Multiple numeric feature groups | Multi-tower bilinear MLP with optional FiLM conditioning |
| `bilinear-fp` | SMILES + numeric features | Bilinear model with chemical fingerprints |
| `chemprop` | SMILES | Message-passing neural network (wraps chemprop v2) |
| `cnn` | Images | 2D convolutional neural network ensemble |

### Training

```python
from duvidnn.autoclass import AutoModelBox

modelbox = AutoModelBox("fingerprint", n_units=16, n_hidden=2, ensemble_size=10)._instance

modelbox.load_training_data(
    data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train",
    structure_column="smiles",
    labels="clogp",
)

modelbox.train(
    val_data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    epochs=10,
    batch_size=128,
)
```

`ModelBox.train()` uses PyTorch Lightning under the hood. You can pass Lightning
callbacks and trainer options via the `callbacks` and `trainer_opts` parameters.

### Saving and loading checkpoints

```python
modelbox.save_checkpoint("checkpoint.dv")

# Later...
from duvidnn.autoclass import AutoModelBox
modelbox = AutoModelBox.from_pretrained("checkpoint.dv")
```

Checkpoints are directories containing model weights (`params.pt`), model
configuration JSON, and the training dataset (as a HuggingFace Dataset on disk).
Checkpoints can also be loaded from HuggingFace Hub repositories using the
`hf://` prefix.

### Prediction and evaluation

```python
# Raw predictions (returns HuggingFace Dataset)
predictions = modelbox.predict(
    data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
)

# Predictions + metrics (returns DataFrame, dict)
predictions_df, metrics = modelbox.evaluate(
    data="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
)
# metrics = {"rmse": ..., "pearson_r": ..., "spearman_rho": ...}
```

### Uncertainty and information metrics

**Ensemble variance** requires `ensemble_size > 1`:

```python
variance = modelbox.prediction_variance(
    candidates="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
)
```

**Doubtscore** measures how much a single test point would shift each model
parameter, normalized by the Fisher score of the training data:

```python
doubtscore = modelbox.doubtscore(
    candidates="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
)
```

**Information sensitivity** additionally accounts for second-order curvature of
the loss landscape:

```python
info_sens = modelbox.information_sensitivity(
    candidates="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    approx="bekas",
    n=10,
)
```

Speed/accuracy trade-offs for information sensitivity:

| Option | Effect |
|--------|--------|
| `last_layer_only=True` | Restrict to output-layer parameters (biggest speed-up) |
| `optimality_approximation=True` | Assume gradient of loss is zero at trained parameters |
| `approx="bekas"` | Stochastic Hessian diagonal approximation |

Results are returned as HuggingFace Datasets, which can be materialized in memory:

```python
doubtscore_df = doubtscore.to_pandas()
```

## Architecture

### Package layout

```
duvidnn/
    __init__.py                 # Version, app name
    autoclass.py                # AutoModelBox factory and MODELBOX_REGISTRY
    checkpoint_utils.py         # Save/load JSON, HF datasets, PyTorch weights
    hyperparameters.py          # Hyperparameter grid generation

    base/                       # Framework-agnostic base classes
        modelboxes.py           # ModelBoxBase, ModelBoxWithVarianceBase,
                                #   FingerprintModelBoxBase, ChempropModelBoxBase
        modelbox_registry.py    # @register_modelbox decorator, MODELBOX_REGISTRY
        information.py          # DoubtMixinBase (doubtscore, info sensitivity logic)
        data.py                 # DataMixinBase, ChemMixinBase (data loading, ingestion)
        training.py             # ModelTrainerBase (abstract trainer)
        evaluation.py           # rmse, mae, pearson_r, spearman_r
        aggregators.py          # Aggregator functions (mean, var, rms, etc.)
        typing.py               # Type aliases (DataLike, FeatureLike, etc.)
        preprocessing/
            registry.py         # @register_function decorator, FUNCTION_REGISTRY
            serializing.py      # Preprocessor (JSON-serializable featurizer config)
            functions.py        # Built-in featurizers: Identity, Log, OneHot, Hash,
                                #   MorganFingerprint, Descriptors2D, VectomeFingerprint,
                                #   ChempropData
            deep_functions.py   # HfBART (HuggingFace transformer featurizer)

    torch/                      # PyTorch implementations
        functions.py            # Loss functions (mse_loss, mae_loss, cosine_loss)
        models/
            mlp.py              # TorchMLPBase, TorchMLPLightning, TorchMLPEnsemble
            bilinear.py         # TorchBilinearBase, TorchBilinearEnsemble
            cnn.py              # TorchCNN2DEnsemble
            chemprop/           # Chemprop v2 integration
            utils/
                ensemble.py     # TorchEnsembleMixin
                lt.py           # LightningMixin
        modelbox/
            modelboxes.py       # Concrete ModelBox classes (TorchMLPModelBox, etc.)
            data.py             # DataMixin, ChempropDataMixin
            information.py      # DoubtMixin, ChempropDoubtMixin (PyTorch grad logic)
            training.py         # ModelTrainer (Lightning Trainer wrapper)

    cli_module/                 # Command-line interface
        cli.py                  # Argument definitions and CLIApp setup
        train.py                # Training entrypoint
        predict.py              # Prediction entrypoint
        split.py                # Data splitting entrypoint
        percentile.py           # Percentile annotation entrypoint
        hyperprep.py            # Hyperparameter grid entrypoint
        eval.py                 # Evaluation helper (not yet registered as CLI command)
        io.py                   # I/O helpers
        utils.py                # CLI utility functions

    utils/                      # Shared utilities
        splitting/              # Scaffold, FAISS, and bin-packing splitters
        datasets.py             # Dataset loading helpers
        lightning.py            # Lightning utility functions
        plotting.py             # Plotting helpers
        package_data.py         # Cache directory management
```

### Class hierarchy

The `ModelBox` hierarchy separates concerns into composable mixins:

```
ModelBoxBase (base/modelboxes.py)
    Inherits: DataMixinBase, DoubtMixinBase, ABC
    Provides: train, predict, evaluate, save/load_checkpoint, create_model (abstract)
    |
    +-- ModelBoxWithVarianceBase
    |       Adds: prediction_variance (ensemble variance)
    |       |
    |       +-- TorchModelBoxBase (torch/modelbox/modelboxes.py)
    |       |       Mixes in: DataMixin, DoubtMixin
    |       |       Adds: save/load_weights, eval_mode, create_trainer
    |       |       |
    |       |       +-- TorchMLPModelBox          ("mlp")
    |       |       +-- TorchBilinearModelBox      ("bilinear")
    |       |       +-- TorchCNN2DModelBox         ("cnn")
    |       |
    |       +-- FingerprintModelBoxBase
    |       |       Mixes in: ChemMixinBase
    |       |       Adds: chemical featurizer construction, SMILES handling
    |       |       |
    |       |       +-- TorchFingerprintModelBox   ("fingerprint")
    |       |       +-- TorchBilinearFingerprintModelBox ("bilinear-fp")
    |       |
    |       +-- ChempropModelBoxBase
    |               Adds: Chemprop data pipeline integration
    |               |
    |               +-- ChempropModelBox           ("chemprop")
```

Each concrete `ModelBox` is decorated with `@register_modelbox("name")`, which
adds it to `MODELBOX_REGISTRY` and makes it available via `AutoModelBox`.

### Preprocessing pipeline

Input features are transformed through a pipeline of registered preprocessing
functions. Each function is registered with the `@register_function("name")`
decorator and returns a callable that maps `(data_dict, input_column) -> ndarray`.

Built-in preprocessors:

| Name | Extras needed | Description |
|------|---------------|-------------|
| `identity` | - | Pass-through |
| `log` | - | Log10 transform |
| `one-hot` | - | One-hot encoding of categorical labels |
| `hash` | - | Deterministic string hashing to fixed-length vectors |
| `morgan-fingerprint` | `chem` | Morgan fingerprint from SMILES (via schemist) |
| `descriptors-2d` | `chem` | 2D molecular descriptors (via schemist) |
| `vectome-fingerprint` | `bio` | Taxonomic embedding from species name (via vectome) |
| `chemprop-mol` | `chem` | Chemprop MoleculeDatapoint from SMILES |
| `hf-bart` | `transformers` | HuggingFace BART encoder features |

Preprocessors are serializable: a `Preprocessor` wraps a function name and its
kwargs into a JSON-compatible dict, so preprocessing pipelines are fully
reproducible and saved alongside checkpoints.

### Information metric internals

Doubtscore and information sensitivity are computed in three stages:

1. **Fisher score** over the training set: for each model parameter, compute
   the gradient of the loss with respect to that parameter, averaged across all
   training examples. For information sensitivity, the diagonal of the Fisher
   information matrix (Hessian of the loss) is also computed.

2. **Parameter gradient** (and optionally Hessian diagonal) for each candidate
   test point.

3. **Score computation**: doubtscore divides the candidate's parameter gradient
   by the training Fisher score. Information sensitivity additionally uses the
   second-order terms.

The heavy lifting is done by the [duvida](https://github.com/scbirlab/duvida)
library, which provides `vmap`-based functional gradient and Hessian
computations via `torch.func`. The `DoubtMixin` class
(`torch/modelbox/information.py`) creates stateless model closures using
`torch.func.functional_call`, then passes them to duvida's `fisher_score`,
`parameter_gradient`, `fisher_information_diagonal`, and
`parameter_hessian_diagonal` functions.

For models with non-standard forward passes (e.g. Chemprop), the `_unrolled`
variants of gradient functions are used, which iterate per-example rather than
vectorizing with `vmap`.

## Extending duvidnn

### Adding a new ModelBox

Subclass `TorchModelBoxBase` (or `FingerprintModelBoxBase` for chemistry models)
and implement `create_model()`:

```python
from duvidnn.torch.modelbox import TorchModelBoxBase
from duvidnn.base.modelbox_registry import register_modelbox
from duvidnn.base.modelboxes import ModelBoxWithVarianceBase

@register_modelbox("my-model")
class MyModelBox(TorchModelBoxBase, ModelBoxWithVarianceBase):

    def create_model(self, *args, **kwargs):
        self._model_config.update(kwargs)
        return MyEnsembleModel(
            n_input=self.input_shape[-1],
            n_out=self.output_shape[-1],
            **self._model_config,
        )
```

Your model class should mix in `TorchEnsembleMixin` (for ensemble support) and
`LightningMixin` (for training). See `duvidnn/torch/models/mlp.py` for a
complete example.

If your data needs preprocessing, implement a static `preprocess_data()` method
on the ModelBox.

### Adding a new preprocessing function

Use the `@register_function` decorator:

```python
from duvidnn.base.preprocessing.registry import register_function

@register_function("my-featurizer")
def MyFeaturizer(some_param: int = 10):
    def _featurize(data, input_column):
        import numpy as np
        return np.stack([transform(v, some_param) for v in data[input_column]])
    return _featurize
```

The function will be available by name in preprocessing pipelines and
serializable into checkpoint configs.

## Known issues

| Issue | Severity | Location |
|-------|----------|----------|
| `cosine_loss` calls `torch.cos(y_pred, y_true)` but `torch.cos` is unary; should use `torch.nn.functional.cosine_similarity` | Critical | `torch/functions.py:98` |
| `"rms"` aggregator uses L2 norm (`np.linalg.norm(ord=2)`), not root-mean-square (`sqrt(mean(x**2))`) | Medium | `base/aggregators.py:17` |
| `eval` subcommand exists in code but is not registered in the CLI app | Medium | `cli_module/eval.py`, `cli_module/cli.py:514-520` |
| `training_dataset` parameter in `_info_score_entrypoint` is accepted but never used; the method always falls back to `self._check_training_data()` | Medium | `base/information.py:365` |
| Batch normalization breaks doubtscore and information sensitivity calculations (chemprop models) | Medium | `torch/models/chemprop/models.py:66-68` |

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/duvidnn/issues).
