# Integrating diagenetic equations using Python

This repo was created as an attempt to reproduce the plots shown at the kickoff of the AstroTOM ("Turing or Milankovitch") project by Niklas Hohmann, from his Matlab scripts. 

AstroTOM is an OpenSSI 2021b project from the Netherlands eScience Center and Utrecht University (UU).

Dr. Emilia Jarochowska (UU) is the lead applicant of this project.

After replacing central differencing for the gradients in the five diagenetic equations from l'Heureux (2018) by forward and backward differencing depending on the sign of U and W as a first step and a Fiadeiro-Veronis spatial difference scheme as a second step, it turns out that these equations can be integrated for more than 13.190 years (the full Tstar) with an implicit or explicit (in time) solver, but not with a simple Eulerian scheme. A Runge-Kutta solver, with an adaptive timestep will, however, suffice.
Implicit (in time) solvers with use of Jacobians are available in the `Use_solve_ivp_without_py-pde_wrapper` branch.

Wide use is made of the [py-pde](https://py-pde.readthedocs.io/en/latest/) package.

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
pipenv shell
```

For the latter command you need `pipenv` which you can install
using either
`pip install pipenv`
or
`conda install -c conda-forge pipenv`.

After `pipenv shell` you should be able to execute

```
python ScenarioA.py
```

After ten minutes you should see plots similar to figure 3e from l'Heureux (2018).

To be done:
1) Read constants from a config file
2) Fix Numpy backend, now only the Numba backend yields correct results.
3) Make plots as nice as those from Use_solve_ivp_without_py-pde_wrapper branch.
