# 🧐 duvida

![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/scbirlab/duvida/python-publish.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/duvida)
![PyPI](https://img.shields.io/pypi/v/duvida)

**duvida** (Portuguese for _doubt_) is a suite of python tools for calculating confidence and information metrics 
for deep learning. It provides lower-level function transforms for exact and approximate Hessian diagonals 
in JAX and pytorch, as well as a higher-level framework for calculating confidence and information metrics
of geenral purpose and chemistry-specific neural networks. 

As a bonus, **duvida** also provides an easy command-line interface for training and testing models.

- [Installation](#installation)
- [Python API](#python-api)
    - [Neural networks](#neural-networks)
    - [Exact and approximate Hessian diagonals](#exact-and-approximate-hessian-diagonals)
- [More advanced API](#more-advanced-python-api-implementing-a-new-modelbox)
- [Command-line interface](#command-line-interface)
- [Issues, problems, suggestions](#issues-problems-suggestions)
- [Documentation](#documentation)

## Installation

### The easy way

You can install the precompiled version directly using `pip`. You need to specify the machine learning framework
that you want to use:

```bash
$ pip install duvida[jax]
# or
$ pip install duvida[jax_cuda12]  # for JAX installing CUDA 12 for GPU support
# or
$ pip install duvida[jax_cuda12_local]  # for JAX using a locally-installed CUDA 12
# or
$ pip install duvida[torch]
```

If you want to use duvida for chemistry machine learning and AI (using the pytorch backend), use:

```bash
$ pip install duvida[chem]
```

We have implemented JAX and pytorch functional transformations for approximate and exact Hessian diagonals,
and doubtscore and information sensitivity. These can be used with JAX- and pytorch-based frameworks.

At the moment, training and inference of full models in `ModelBox` objects is implemented only in pytorch. 

### From source

Clone the repository, then `cd` into it. Then run:

```bash
$ pip install -e .[torch]
```

## Python API

### Neural networks

The core of **duvida** is the `ModelBox`, which is a container for a trainable model and its training data.
These are connected because measures of confidence and information gain depend directly on the information
or evidence already seen by the model.

There are several `ModelBox` classes for specific deep learning architechtures in pytorch. 

```python
>>> from duvida.torch.models import _MODEL_CLASSES
>>> from pprint import pprint
>>> pprint(_MODEL_CLASSES)
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

### Exact and approximate Hessian diagonals

**duvida** provides functional transforms for JAX and pytorch that calculate 
either exact or approximate Hessian diagonals.

You can check which backend you're using:

```python
>>> from duvida.stateless.config import config
>>> config
Config(backend='jax', precision='double', fallback=True)
```

It can be changed:

```python
>>> config.set_backend("torch")
'torch'
>>> config
Config(backend='torch', precision='double', fallback=True)
```

Now you can calculate exact Hessian diagonals without calculating the 
full matrix:

```python
>>> from duvida.stateless.utils import hessian
>>> import duvida.stateless.numpy as dnp 
>>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
>>> a = dnp.array([1., 2.])
>>> exact_diagonal(f)(a) == dnp.diag(hessian(f)(a))
Array([ True,  True], dtype=bool)
```

Various approximations are also allowed.

```python
>>> from duvida.stateless.hessians import get_approximators
>>> get_approximators()  # No arguments to list available
('squared_jacobian', 'exact_diagonal', 'bekas', 'rough_finite_difference')
```

Now apply:

```python
>>> approx_hessian_diag = get_approximators("bekas")
>>> g = lambda x: dnp.sum(dnp.sum(x) ** 3. + x ** 2. + 4.)
>>> a = dnp.array([1., 2.])
>>> dnp.diag(hessian(g)(a))  # Exact for reference
Array([38., 38.], dtype=float64)
>>> approx_hessian_diag(g, n=1000)(a)  # Less accurate when parameters interact
Array([38.52438307, 38.49679655], dtype=float64)
>>> approx_hessian_diag(g, n=1000, seed=1)(a)  # Change the seed to alter the outcome
Array([39.07878869, 38.97796601], dtype=float64)
```

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

Then subclass `duvida.torch.nn.ModelBox` and implement the `create_model()` method, which should
simply return your instantiated model. If you want to preprocess input data on the fly, then
add a `preprocess_data()` method which takes a data dictionary and returns a data dictionary.

```python
from typing import Dict

from duvida.torch.nn import ModelBox
import numpy as np

class MLPModelBox(ModelBox):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._mlp_kwargs = kwargs

    def create_model(self, *args, **kwargs):
        return SimpleMLP(
            n_input=self.input_shape[-1],
            n_out=self.output_shape[-1], 
            *args, **kwargs,
            **self._mlp_kwargs,
        )

    # Define this method if your data needs preprocessing
    @staticmethod
    def preprocess_data(data: Dict[str, np.ndarray], _in_key, _out_key, **kwargs) -> Dict[str, np.ndarray]:
        return {
            _in_key: your_featurizer(data[_in_key]), 
            _out_key: np.asarray(data[_out_key])
        }
```

If the built-in `ModelBox`es don't suit your needs, you can subclass the `base_classes.ModelBoxBase` abstract 
class, making sure to implement its abstract methods.

## Command-line interface

**duvida** has a command-line interface for training and checkpointing the built-in models. 

```bash
$ duvida --help
```

To train:

```bash
$ duvida train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test --ensemble-size 10 --epochs 10 --learning-rate 0.001
```

You can read about all the options here:

```bash
$ duvida train --help
```

There is also a simple hyperparameter utility.

```bash
$ printf '{"model_class": "fingerprint", use_2d": [true, false], "n_units": 16, "n_hidden": 3}' | duvida hyperprep -o hyperopt.json
```

This generates a file containing the Cartesian product of the JSON items. It can be indexed (0-based) 
with the `-i <int>` option to supply a specific training configuration like so:

```bash
$ duvida train hf://scbirlab/fang-2023-biogen-adme@scaffold-split:train -2 hf://scbirlab/fang-2023-biogen-adme@scaffold-split:test -c hyperopt.json -i 0
```

## Issues, problems, suggestions

Add to the [issue tracker](https://www.github.com/scbirlab/duvida/issues).

## Documentation

(To come at [ReadTheDocs](https://duvida.readthedocs.org).)