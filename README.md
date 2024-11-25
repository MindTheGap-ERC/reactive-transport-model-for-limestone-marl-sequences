# Reactive transport model for limestone-marl sequences

This repo was forked off [Integrating-diagenetic-equations-using-Python](https://github.com/astro-turing/Integrating-diagenetic-equations-using-Python), which is part of the [Astro-turing organisation](https://github.com/astro-turing), created as part of the AstroToM ("Turing or Milankovitch") project, an OpenSSI 2021b project from the Netherlands eScience Center and Utrecht University (UU). 

The diagenetic modelling efforts within AstoToM can be regarded as a precursor to this repo and to [rhythmite](https://github.com/MindTheGap-ERC/rhythmite), while [LMA_utils](https://github.com/MindTheGap-ERC/LMA_utils), [LHeureuxEqs](https://github.com/MindTheGap-ERC/LHeureuxEqs) and [Cross-comparison](https://github.com/MindTheGap-ERC/Cross-comparison) are auxiliary repos. 

[LMA-Matlab](https://github.com/MindTheGap-ERC/LMA-Matlab) was the first attempt to reproduce the results from [L'Heureux (2018)](https://onlinelibrary.wiley.com/doi/10.1155/2018/4968315). That repo is coded in MATLAB, while the [original diagenetic model from L'Heureux](https://github.com/astro-turing/Diagenetic_model_LHeureux_2018) was written in FORTRAN. 
[Integrating-diagenetic-equations-using-Python](https://github.com/astro-turing/Integrating-diagenetic-equations-using-Python) was inspired by [LMA-Matlab](https://github.com/MindTheGap-ERC/LMA-Matlab).

MindTheGap is led by dr. Emilia Jarochowska (UU). 

Wide use is made of the [py-pde](https://py-pde.readthedocs.io/en/latest/) package.

The porosity diffusion coefficient is held constant.

## Installing and using
To run this code, you need `git` and `conda` or `pip` to install .
```
git clone git@github.com:MindTheGap-ERC/reactive-transport-model-for-limestone-marl-sequences.git
```
or 
```
git clone https://github.com/MindTheGap-ERC/reactive-transport-model-for-limestone-marl-sequences.git
```
Next,
```
cd reactive-transport-model-for-limestone-marl-sequences
git switch release_v1.0.0
pipenv install
```

For the latter command you need `pipenv` which you can install
using either
`pip install pipenv`
or
`conda install -c conda-forge pipenv`.

Now you may be running into certain Python version requirements, i.e. the Pipfile requires a Python version that you do not have installed. For this conda can help, e.g.:
`conda create -n py311 python=3.11 anaconda` to create a Conda Python 3.11 environment. 

You can use that freshly installed Python version and possibly any additionally installed libraries - using the `--site-packages` argument - by executing `pipenv install --python=$(conda run -n py311 which python) --site-packages --skip-lock`. The latter argument - `--skip-lock` - may be redundant, but if your previous `pipenv install` failed, `pipenv --rm` may be needed. 

After a succesful pipenv installation you should be able to execute

```
pipenv run python marlpde/Evolve_scenario.py
```
or

```
pipenv shell
python marlpde/Evolve_scenario.py
```
Results in the form of an .hdf5 file will be stored in a subdirectory of a `Results` directory, which will be in the root folder of the cloned repo.

### Alternative: poetry
If you prefer [`poetry`](https://python-poetry.org/) over `pipenv`, you may install all the dependencies and activate the environment using the command `poetry install`. Next, either:

```
poetry run python marlpde/Evolve_scenario.py
```
or

```
poetry shell
python marlpde/Evolve_scenario.py
```

## Running tests

From the root folder, i.e. the folder you enter after `cd Integrating-diagenetic-equations-using-Python`, either run
```
pipenv run python -m pytest
```
or

```
poetry run python -m pytest
```

## Copyright

Copyright 2023 Netherlands eScience Center and Utrecht University

## Funding information
Funded by the European Union (ERC, MindTheGap, StG project no 101041077). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them.
![European Union and European Research Council logos](https://erc.europa.eu/sites/default/files/2023-06/LOGO_ERC-FLAG_FP.png)
