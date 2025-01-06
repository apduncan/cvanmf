# Installation

The package can be installed from pypi via `pip`. In the future we intend to provide a `conda`/`mamba` install. 

Our current suggestion for installation is to use `mamba` to manage the python version, and pip to install `cvanmf`, 
as shown below.

## pip install
First make a new environment with python 3.12 or higher

```commandline
mamba create --name cvanmf 'python>=3.12' pip
```

(If you use conda, substitute `mamba` for `conda` at the start). Once this has 
completed, activate the environment and install `cvanmf` which will also 
install it's dependencies.

```commandline
mamba activate cvanmf
pip install cvanmf
```

Test the installation by running one of the commands, which should now give 
you a brief message

```commandline
rank_select

Usage: rank_select [OPTIONS]
Try 'rank_select --help' for help.

Error: Missing option '-i' / '--input'.
```

### Issues with `gcc`
Some packages which `cvanmf` depends upon require `gcc` to install, which is not available on some systems by default.
If using `conda`/`mamba` you can install it in the local environment
```
mamba install --name cvanmf gcc
```
or use the system package manager for to install (e.g. `sudo apt install gcc` for debian based distros).

## Updating

To update to a new version of `cvanmf`, with your environment active, run

```commandline
pip --upgrade cvanmf
```

## Additional installation: Jupyter
Many examples in this documentation are written as Jupyter notebooks, and 
this can be a convenient way to use the package. For an introduction to 
Jupyter Lab [see here](https://jupyterlab.readthedocs.io/en/stable/index.html).

To install Jupyter Lab

```commandline
mamba install --name cvanmf jupyterlab
```

When installed, to launch Jupyter lab, with the `cvanmf` environment active, run

```
jupyter lab
```