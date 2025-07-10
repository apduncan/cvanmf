# Installation

The package can be installed from either bioconda, or via `pip`.

## conda install
Our recommendation is to install cvanmf from bioconda.
To do this you should have `conda`, or one of the reimplementations (`mamba`/`micromamba`), installed.
The example below will use `conda` but the install commands are the same for `mamba` or `micromamba`.

To install in a new environment, run the command below, replacing `{env_name}` with the name you want for the
environment.

```
conda install --name {env_name} -c bioconda -c conda-forge cvanmf
```

## pip install
To install via pip, you will need a version of python >=3.12.
Run:

```
pip install cvanmf
```

## Test installation
Test the installation by running one of the commands, which should now give 
you a brief message

```commandline
rank_select

Usage: rank_select [OPTIONS]
Try 'rank_select --help' for help.

Error: Missing option '-i' / '--input'.
```

## Troubleshooting
### Issues with `gcc`
Some packages which `cvanmf` depends upon require `gcc` to install, which is not available on some systems by default.
If using `conda`/`mamba` you can install it in the local environment
```
mamba install --name cvanmf gcc
```
or use the system package manager for to install (e.g. `sudo apt install gcc` for debian based distros).

## Additional installation: Jupyter
Many examples in this documentation are written as Jupyter notebooks, and 
this can be a convenient way to use the package. For an introduction to 
Jupyter Lab [see here](https://jupyterlab.readthedocs.io/en/stable/index.html).

To install Jupyter Lab

```
conda install --name {env_name} jupyterlab
```

When installed, to launch Jupyter lab, with the `cvanmf` environment active, run

```
jupyter lab
```