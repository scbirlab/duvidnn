# 🧐 duvida

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/duvidnn/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/duvidnn)
![PyPI](https://img.shields.io/pypi/v/duvidnn)

**duvidnn** is a suite of python tools for calculating confidence and information metrics 
for deep learning. It provides a higher-level framework for calculating confidence and information metrics
of general purpose, taxonomic and chemistry-specific neural networks. 

As a bonus, **duvida** also provides an easy command-line interface for training and testing models.

- [Installation](#installation)
- [Python API](#python-api)
    - [Neural networks](#neural-networks)
- [More advanced API](#more-advanced-python-api-implementing-a-new-modelbox)
- [Command-line interface](#command-line-interface)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)

## Installation

### The easy way

You can install the precompiled version directly using `pip`.

```bash
$ pip install duvidnn
```

If you want to use duvida for chemistry machine learning and AI, use:

```bash
$ pip install duvida[chem]
```

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .
```

## Python API

### Neural networks

The core of **duvida** is the `ModelBox`, which is a container for a trainable model and its training data.
These are connected because measures of confidence and information gain depend directly on the information
or evidence already seen by the model.

There are several `ModelBox` classes for specific deep learning architechtures in pytorch. 

```python
>>> from duvida.base.modelbox_registry import MODELBOX_REGISTRY
>>> from pprint import pprint
>>> pprint(MODELBOX_REGISTRY)
{'chemprop': <class 'duvida.torch.chem.ChempropModelBox'>,
 'fingerprint': <class 'duvida.torch.chem.FPMLPModelBox'>,
 'fp': <class 'duvida.torch.chem.FPMLPModelBox'>,
 'mlp': <class 'duvida.torch.models.mlp.MLPModelBox'>}
```

The modelboxes `chemprop` and `fingerprint` (alias `fp`) featurize SMILES representations of chemical 
structures. The modelbox `mlp` is a general purpose multilayer perceptron.

You can set up your model with various training parameters.

```python
from duvida.autoclass import AutoClass
modelbox = AutoClass(
    "fingerprint",
    n_units=16,
    n_hidden=2,
    ensemble_size=10,
)
```

The internal neural network is instantiated on loading training data.

```python
modelbox.load_training_data(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train",
    inputs="smiles",
    labels="clogp",
)
```

The `filename` can be a Huggingface dataset, in which case it is automatically downloaded. The `"@"`
indicates the dataset configuration, and the `":"` indicates the specific data split.

Alternatively, the training data can be a local CSV or TSV file. In-memory Pandas dataframes 
or dictionaries can be supplied through the `data` argument.

With training data loaded, the model can be trained!

```python
modelbox.train(
    val_filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    epochs=10,
    batch_size=128,
)
```

The `ModelBox.train()` method uses pytorch Lightning under the hood, so other options such as callbacks
for this framework should be accepted.

#### Saving and sharing a trained model

**duvida** provides a basic checkpointing mechanism to save model weights and training data to later reload.

```python
modelbox.save_checkpoint("checkpoint.dv")
modelbox.load_checkpoint("checkpoint.dv")
```

#### Evaluating and predicting on new data

**duvida** `ModelBox`es provide methods for evaluating predictions on new data.

```python
from duvida.evaluation import rmse, pearson_r, spearman_r
predictions, metrics = modelbox.evaluate(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    metrics={
        "RMSE": rmse, 
        "Pearson r": pearson_r, 
        "Spearman rho": spearman_r
    },
)
```

#### Calculating uncertainty and information metrics

**duvida** `ModelBox`es provide methods for calculating prediction variance of ensembles,
doubtscore, and information sensitivity.

```python
doubtscore = modelbox.doubtscore(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test"
)
info_sens = modelbox.information_sensitivity(
    filename="hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test",
    approx="bekas",  # approximate Hessian diagonals
    n=10,
)
```

To avoid storing large datasets in memory, **duvida** uses Huggingface datasets under the hood
to cache data. Results can be instantiated in memory with a little effort. For example:

```python
doubtscore = doubtscore.to_pandas()
```

See the [Huggingface datasets documentation](https://huggingface.co/docs/datasets/) for more.

## More advanced Python API: Implementing a new `ModelBox`

Bringing a new pytorch model to **duvida** is relatively straightforward. First, write your model,
adding Lighning logic and a `create_model()` method:

```python
from typing import Callable, Iterable, List, Mapping, Optional

from torch.nn import BatchNorm1d, Dropout, Linear, Module, SiLU, Sequential
from duvida.torch.models.ensemble import TorchEnsembleMixin
from duvida.torch.models.lt import LightningMixin
from torch.nn import Module
from torch.optim import Adam, Optimizer

class SimpleMLP(torch.nn.Module, LightningMixin):

    def __init__(
        self, 
        n_input: int, 
        n_units: int = 16, 
        n_out: int = 1,
        activation: Callable = torch.nn.SiLU,  # Smooth activation to prevent vanishing gradient
        learning_rate: float = .01,
        optimizer: Optimizer = Adam,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.n_input = n_input
        self.n_units = n_units
        self.activation = activation
        self.n_out = n_out
        self.model_layers = torch.nn.Sequential([
            torch.nn.Linear(self.n_input, self.n_units),
            self.activation(),
            torch.nn.Linear(self.n_units, self.n_out),
        ])
        # Lightning logic
        self._init_lightning(
            optimizer=optimizer, 
            learning_rate=learning_rate, 
            model_attr='model_layers',  # the attribute containing the model
        )

    def forward(self, x):
        return self.model_layers(x)
```

Then subclass `duvida.torch.modelbox.TorchModelBoxBase` and implement the `create_model()` method, which should
simply return your instantiated model. If you want to preprocess input data on the fly, then
add a `preprocess_data()` method which takes a data dictionary and returns a data dictionary.

```python
from typing import Dict

from duvida.torch.modelbox import TorchModelBoxBase
import numpy as np

class MLPModelBox(TorchModelBoxBase):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._mlp_kwargs = kwargs

    def create_model(self, *args, **kwargs):
        self._model_config.update(kwargs)  # makes sure model checkpointing saves the keyword args
        return SimpleMLP(
            n_input=self.input_shape[-1],  # defined on data loading
            n_out=self.output_shape[-1], 
            *args, 
            **self._model_config,
            **self._mlp_kwargs,  # if init kwargs are relevant to model creation
        )

    # Define this method if your data needs preprocessing
    @staticmethod
    def preprocess_data(data: Dict[str, np.ndarray], _in_key, _out_key, **kwargs) -> Dict[str, np.ndarray]:
        return {
            _in_key: your_featurizer(data[_in_key]), 
            _out_key: np.asarray(data[_out_key])
        }
```

If you want to build `ModelBox`es based on a framework other than pytorch, you can subclass 
the `duvida.base.ModelBoxBase` abstract class, making sure to implement its abstract methods.

## Command-line interface

**duvidnn** has a command-line interface for training and checkpointing the built-in models. 

```bash
$ duvida --help
usage: duvidnn [-h] [--version] {hyperprep,train,predict,split,percentiles} ...

Calculating exact and approximate confidence and information metrics for deep learning on general purpose and chemistry tasks.

options:
  -h, --help            show this help message and exit
  --version, -v         show program's version number and exit

Sub-commands:
  {hyperprep,train,predict,split,percentiles}
                        Use these commands to specify the tool you want to use.
    hyperprep           Prepare inputs for hyperparameter search.
    train               Train a PyTorch model.
    predict             Make predictions and calculate uncertainty using a duvida checkpoint.
    split               Make chemical train-test-val splits on out-of-core datasets.
    percentiles         Add columns indicating whether rows are in a percentile.
```

In all cases, you can get further options with `duvida <command> --help`, for example:

```bash
duvida train --help
```

### Annotating top percentiles

You can add columns to datasets which annotate the top percentiles of named columns. This is compatible
with datasets that don't fit in memory.

```bash
$ duvidnn percentiles \
    hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --columns clogp tpsa \
    --percentiles 1 5 10 \
    --output percentiles.parquet \
    --plot percentiles-plot.png \
    --structure smiles
```

### Data splitting

There are utilities for out-of-memory scaffold and (approximate using FAISS) spectral splitting of datasets
that don't fit in memory. Make it random but reproducible with `--seed`, otherwise a deterministic bin-packing
algorithm is used.

```bash
$ duvidnn split hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    --train .7 \
    --validation .15 \
    --structure smiles \
    --type faiss \
    --seed 1 \
    --output faiss.csv \
    --plot faiss.png
  ```

### Model training and evaluation

To train:

```bash
$ duvidnn train -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --ensemble-size 10 --epochs 10 --learning-rate 0.001 \
    --output model.dv
```

### Hyperparameters

There is also a simple hyperparameter utility.

```bash
$ printf '{"model_class": "fingerprint", use_2d": [true, false], "n_units": 16, "n_hidden": 3}' | duvidnn hyperprep -o hyperopt.json
```

This generates a file containing the Cartesian product of the JSON items. It can be indexed (0-based) 
with the `-i <int>` option to supply a specific training configuration like so:

```bash
$ duvidnn train \
    -1 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train \
    -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    -c hyperopt.json \
    -i 0 \
    --output model.dv
```

This **overrides any conflicting command line arguments**.

### Predictions

You cna make predictions on datasets using `duvida predict`, and optionally predict only a chunk of the dataset using `--start` and `--stop`, in case you
want to parallelize.

When predicting, there is also the option to calculate uncertainty metrics like ensemble variance, Tanomoto nearest neighbor distance to training set
(for chemistry models), doubtscore, and information sensitivity.

```bash
$ duvidnn predict \
    --test hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test \
    --checkpoint model.dv \
    --start 100 \
    --end 200 \
    --variance \
    --tanimoto \
    --doubtscore \
    -y clogp \
    --output predictions.parquet
```

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/duvidnn/issues).

## Documentation

(To come at [ReadTheDocs](https://duvida.readthedocs.org).)