# Installation

Currently the package is only installable via `pip` from github. In the 
future I intend to provide a `conda`/`mamba` install. Currently I use 
`mamba` to manage the python version, and pip to install `cvanmf`, as shown 
below.

## pip install
First make a new environment with python 3.10 or higher

```commandline
mamba create --name cvanmf 'python>=3.10'
```

(If you use conda, substitute `mamba` for `conda` at the start). Once this has 
completed, activate the environment and install `cvanmf` which will also 
install it's dependencies.

```commandline
mamba activate cvanmf
pip install git+https://github.com/apduncan/cvanmf.git
```

Test the installation by running one of the commands, which should now give 
you a brief message

```commandline
rank_select

Usage: rank_select [OPTIONS]
Try 'rank_select --help' for help.

Error: Missing option '-i' / '--input'.
```

## Additional installation: Jupyter
Many examples in this documentation are written as Jupyter notebooks, and 
this can be a convenient way to use the package. For an introduction to 
Jupyter see here.

To install Jupyter

```commandline
mamba install --name cvanmf jupyter
```

When installed, to launch Jupyer lab, with the `cvanmf` environment active, run

```
jupyter lab
```

## Additional installation: Nextflow pipeline

Todo Nextflow docs.