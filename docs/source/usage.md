# Usage

## Fast and flexible reading and random access of very large files

Subsets of lines from very large, optionally compressed, files can be read quickly 
into memory. for example, we can read the first 10,000 lines of an arbitrarily large 
file:

```python
>>> from carabiner.io import get_lines

>>> get_lines("big-table.tsv.gz", lines=10_000)
```

Or random access of specific lines. Hundreds of millions of lines can be 
parsed per minute.

```python
>>> get_lines("big-table.tsv.gz", lines=[999999, 10000000, 100000001])
```

This pattern will allow sampling a random subset:

```python
>>> from random import sample
>>> from carabiner.io import count_lines, get_lines

>>> number_of_lines = count_lines("big-table.tsv.gz")
>>> line_sample = sample(range(number_of_lines), k=1000)
>>> get_lines("big-table.tsv.gz", lines=line_sample)
```

### Reading tabular data

With this backend, we can read subsets of very large files more quickly 
and flexibly than plain `pandas.read_csv`. Formats (delimiters) including Excel 
are inferred from file extensions, but can also be over-ridden with the `format` 
parameter.

```python
>>> from carabiner.pd import read_table

>>> read_table("big-table.tsv.gz", lines=10_000)
```

The same fast random access is availavble as for reading lines. Hundreds of 
millions of records can be looped through per minute.

```python
>>> from random import sample
>>> from carabiner.io import count_lines, get_lines

>>> number_of_lines = count_lines("big-table.tsv.gz")
>>> line_sample = sample(range(number_of_lines), k=1000)
>>> read_table("big-table.tsv.gz", lines=line_sample)
```

## Utilities to simplify building command-line apps

The standard library `argparse` is robust but verbose when building command-line apps with several sub-commands, each with many options. `carabiner.cliutils` smooths this process. Apps are built by defining `CLIOptions` which are then assigned to `CLICommands` directing the functions to run when called, which then form part of a `CLIApp`.

First define the options:
```python
inputs = CLIOption('inputs',
                    type=str,
                    default=[],
                    nargs='*',
                    help='')
output = CLIOption('--output', '-o', 
                    type=FileType('w'),
                    default=sys.stdout,
                    help='Output file. Default: STDOUT')
formatting = CLIOption('--format', '-f', 
                        type=str,
                        default='TSV',
                        choices=['TSV', 'CSV', 'tsv', 'csv'],
                        help='Format of files. Default: %(default)s')
```

Then the commands:

```python
test = CLICommand("test",
                    description="Test CLI subcommand using Carabiner utilities.",
                    options=[inputs, output, formatting],
                    main=_main)
```

The same options can be assigned to multiple commands if necessary.

Fianlly, define the app and run it:

```python

app = CLIApp("Carabiner", 
             version=__version__,
             description="Test CLI app using Carabiner utilities.",
             commands=[test])

app.run()
```
## Reservoir sampling

If you need to sample a random subset from an iterator of unknown length by looping through only once, you can use this pure python implementation of [reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling).

An important limitation is that while the population to be sampled is not necessarily in memory, the sampled population must fit in memory.

Originally written in [Python Bugs](https://bugs.python.org/issue41311).

Based on [this GitHub Gist](https://gist.github.com/oscarbenjamin/4c1b977181f34414a425f68589e895d1).

```python
>>> from carabiner.random import sample_iter
>>> from string import ascii_letters
>>> from itertools import chain
>>> from random import seed
>>> seed(1)
>>> sample_iter(chain.from_iterable(ascii_letters for _ in range(1000000)), 10)
['X', 'c', 'w', 'q', 'T', 'e', 'u', 'w', 'E', 'h']
>>> seed(1)
>>> sample_iter(chain.from_iterable(ascii_letters for _ in range(1000000)), 10, shuffle_output=False)
['T', 'h', 'u', 'X', 'E', 'e', 'w', 'q', 'c', 'w']

```

## Multikey dictionaries

Conveniently return the values of multiple keys from a dictionary without manually looping.

```python
>>> from carabiner.collections import MultiKeyDict
>>> d = MultiKeyDict(a=1, b=2, c=3)
>>> d
{'a': 1, 'b': 2, 'c': 3}
>>> d['c']
{'c': 3}
>>> d['a', 'b']
{'a': 1, 'b': 2} 
```

## Decorators

`carabiner` provides several decorators to facilitate functional programming.

### Vectorized functions

In scientific programming frameworks like `numpy` we are used to functions which take a scalar or vector and apply to every element. It is occasionally useful to convert functions from arbitrary packages to behave in a vectorized manner on Python iterables.

Scalar functions can be converted to a vectorized form easily using `@vectorize`.

```python
>>> @vectorize
... def vector_adder(x): return x + 1
...
>>> list(vector_adder(range(3)))
[1, 2, 3]
>>> list(vector_adder((4, 5, 6)))
[5, 6, 7]
>>> vector_adder([10])
11
>>> vector_adder(10)
11
```

### Return `None` instead of error

When it is useful for a function to not fail, but have a testable indicator of success, you can wrap in `@return_none_on_error`.

```python
>>> def error_maker(x): raise KeyError
... 
>>> @return_none_on_error
... def error_maker2(x): raise KeyError
... 
>>> @return_none_on_error(exception=ValueError)
... def error_maker3(x): raise KeyError
... 

>>> error_maker('a')  # Causes an error
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File "<stdin>", line 1, in error_maker
KeyError

>>> error_maker2('a')  # Wrapped returns None

>>> error_maker3('a')  # Only catches ValueError
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
File ".../carabiner/decorators.py", line 59, in wrapped_function
    
File "<stdin>", line 2, in error_maker3
KeyError
```

### Decorators with parameters

Sometimes a decorator has optional parameters to control its behavior. It's convenient to use it in the form `@decorator` when you want the default behavior, or `@decorator(*kwargs)` when you want to custmize the behavior. Usually this requires some convoluted code, but this has been packed up into `@decorator_with_params`, to decorate your decorator definitions!

```python
>>> def decor(f, suffix="World"): 
...     return lambda x: f(x + suffix)
...
>>> @decor
... def printer(x): 
...     print(x)
... 

# doesn't work, raises an error!
>>> @decor(suffix="everyone")  
... def printer2(x): 
...     print(x)
... 
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
TypeError: decor() missing 1 required positional argument: 'f'

# decorate the decorator!
>>> @decorator_with_params
... def decor2(f, suffix="World"): 
...     return lambda x: f(x + suffix)
... 

# Now it works!
>>> @decor2(suffix="everyone")  
... def printer3(x): 
...     print(x)
... 

>>> printer("Hello ")
Hello World
>>> printer3("Hello ")
Hello everyone
```

## Colorblind palette

Here's a qualitative palette that's colorblind friendly.

```python
>>> from carabiner import colorblind_palette

>>> colorblind_palette()
('#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311', '#009988', '#BBBBBB', '#000000')

# subsets
>>> colorblind_palette(range(2))
('#EE7733', '#0077BB')
>>> colorblind_palette(slice(3, 6))
('#EE3377', '#CC3311', '#009988')
```

## Grids with sensible defaults in Matplotlib

While `plt.subplots()` is very flexible, it requires many defaults to be defined. Instead, `carabiner.mpl.grid()` generates the `fig, ax` tuple with sensible defaults of a 1x1 grid with panel size 3 and a `constrained` layout.

```python
from carabiner.mpl import grid
fig, ax = grid()  # 1x1 grid
fig, ax = grid(ncol=3)  # 1x3 grid; figsize expands appropriately
fig, ax = grid(ncol=3, nrow=2, sharex=True)  #additional parameters are passed to `plt.subplots()`
```

## Fast indicator matrix x dense matrix multiplication in Tensorflow

If you want to multiply an indicator matrix, i.e. a sparse matrix of zeros and ones with the same number of non-zero entries per row (as in linear models), as part of a Tensorflow model, this pattern will be faster than using `tensorflow.SparseMatrix` if you convert the indicator matrix to a `[n x 1]` matrix providing the index of the non-zero element per row.