# Integrating diagenetic equations using Python

This repo was created as an attempt to reproduce the plots shown at the kickoff of the AstroTOM ("Turing or Milankovitch") project by Niklas Hohmann, from his Matlab scripts (available at [github.com/MindTheGap-ERC/LMA-Matlab](https://github.com/MindTheGap-ERC/LMA-Matlab)). 

AstroTOM is an OpenSSI 2021b project from the Netherlands eScience Center and Utrecht University (UU).

Dr. Emilia Jarochowska (UU) is the lead applicant of this project.

After replacing central differencing for the gradients in the five diagenetic equations 40-43 from [L'Heureux (2018)](https://www.hindawi.com/journals/geofluids/2018/4968315/) by forward and backward differencing depending on the sign of U and W as a first step and a Fiadeiro-Veronis spatial difference scheme as a second step, it turns out that these equations can be integrated for more than 13.190 years (the full T*) with an implicit or explicit (in time) solver, but not with a simple Eulerian scheme. A Runge-Kutta solver, with an adaptive timestep will, however, suffice.
After correcting the value of b (5-->5e-4) it turned out that a stable integration is also possible without a Fiadeiro-Veronis scheme. The `main` branch makes use of a constant porosity diffusion coefficient.

Implicit (in time) solvers with use of Jacobians (in functional forms, so without numerical approximations) are available in the `Use_solve_ivp_without_py-pde_wrapper` branch.

Wide use is made of the [py-pde](https://py-pde.readthedocs.io/en/latest/) package, especially in the `main` branch.

## Installing and using
To run this code, you need `git` and `conda` or `pip` to install .
```
git clone git@github.com:astro-turing/Integrating-diagenetic-equations-using-Python.git
```
or 
```
git clone https://github.com/astro-turing/Integrating-diagenetic-equations-using-Python.git
```
Next,
```
cd Integrating-diagenetic-equations-using-Python
pipenv install
```

For the latter command you need `pipenv` which you can install
using either
`pip install pipenv`
or
`conda install -c conda-forge pipenv`.

Now you may be running into certain Python version requirements, i.e. the Pipfile requires a Python version that you do not have installed. For this conda can help, e.g.:
`conda create -n py311 python=3.11 anaconda` to create a Conda Python 3.11 environment. 

`pytables` is a new dependency that caused problems across various platforms. This can be fixed for pipenv installs on Linux and Mac OS by installing pytables through conda: `conda activate py311; conda install -c anaconda pytables`. Windows users are for now encouraged to use the `poetry install`, see below.

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

After two minutes you should see plots similar to figure 3e from [L'Heureux (2018)](https://www.hindawi.com/journals/geofluids/2018/4968315/).

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


