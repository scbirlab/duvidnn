# Installation

## The easy way

Install the pre-compiled version from GitHub:

```bash
$ pip install carabiner-tools
```

If you want to use the `tensorflow`, `pandas`, or `matplotlib` utilities, these must be installed separately
or together:

```bash
$ pip install carabiner-tools[deep]
# or
$ pip install carabiner-tools[pd]
# or
$ pip install carabiner-tools[mpl]
# or
$ pip install carabiner-tools[all]
```

## From source

Clone the repository, then `cd` into it. Then run:

```bash
pip install -e .
```