#!/usr/bin/env python

from datetime import datetime
import os
from dataclasses import asdict
import inspect
from marlpde.marlpde import Map_Scenario, Solver
from LHeureux_model import LMAHeureuxPorosityDiff
import matplotlib.pyplot as plt
from pde import CartesianGrid, ScalarField, FileStorage
from pde.grids.operators.cartesian import _make_derivative

def integrate_equations(**kwargs):
    '''
    This function retrieves the parameters of the Scenario to be simulated and 
    the solution parameters for the integration. It then integrates the five
    partial differential equations form L'Heureux, stores and returns the 
    solution, to be used for plotting.
    '''

    Xstar = kwargs["Xstar"]
    Tstar = kwargs["Tstar"]
    max_depth = kwargs["max_depth"]
    ShallowLimit = kwargs["ShallowLimit"]
    DeepLimit = kwargs["DeepLimit"]
    CAIni = kwargs["CAIni"]
    CCIni = kwargs["CCIni"]
    cCaIni = kwargs["cCaIni"]
    cCO3Ini = kwargs["cCO3Ini"]
    PhiIni = kwargs["PhiIni"]

    Solver_parameters = Solver()
    Number_of_depths = Solver_parameters.N
    # End_time is in units of Tstar.
    End_time = Solver_parameters.tmax/Tstar
    dt = Solver_parameters.dt

    depths = CartesianGrid([[0, max_depth/Xstar]], [Number_of_depths], periodic=False)
    # We will be needing forward and backward differencing for
    # Fiadeiro-Veronis differentiation.
    depths.register_operator("grad_back", \
        lambda grid: _make_derivative(grid, method="backward"))
    depths.register_operator("grad_forw", \
        lambda grid: _make_derivative(grid, method="forward"))
    
    AragoniteSurface = ScalarField(depths, CAIni)
    CalciteSurface = ScalarField(depths, CCIni)
    CaSurface = ScalarField(depths, cCaIni)
    CO3Surface = ScalarField(depths, cCO3Ini)
    PorSurface = ScalarField(depths, PhiIni)
    
    # I need those two fields for computing coA, which is rather involved.
    # There may be a simpler way of selecting these depths, but I haven't
    # found one yet. For now these two Heaviside step functions.
    not_too_shallow = ScalarField.from_expression(depths,
                      f"heaviside(x-{ShallowLimit/Xstar}, 0)")
    not_too_deep = ScalarField.from_expression(depths,
                   f"heaviside({DeepLimit/Xstar}-x, 0)")    
    
    # Not all keys from kwargs are LMAHeureuxPorosityDiff arguments.
    # Taken from https://stackoverflow.com/questions/334655/passing-a-\
    # dictionary-to-a-function-as-keyword-parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in [p.name for p in 
                      inspect.signature(LMAHeureuxPorosityDiff).parameters.\
                        values()]}

    eq = LMAHeureuxPorosityDiff(AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface, not_too_shallow, 
                                not_too_deep, **filtered_kwargs)             
    
    state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                         CO3Surface, PorSurface)
    
    # Store your results somewhere in a subdirectory of a parent directory.
    store_folder = "../Results/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
    os.makedirs(store_folder)
    stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
    storage = FileStorage(stored_results)
    
    sol, info = eq.solve(state, t_range=End_time, dt=dt, method="explicit", \
                   scheme = "rk", tracker=["progress", storage.tracker(0.01)], \
                   backend = "numba", ret_info = True, adaptive = True)
    print()
    print(f"Meta-information about the solution : {info}")        

    covered_time = Tstar * End_time

    return sol, covered_time, depths, Xstar

def Plot_results(sol, covered_time, depths, Xstar):
    '''
    Plot the five fields at the end of the integration interval as a function
    of depth.
    '''
    plt.title(f"Situation after {covered_time:.2f} years")
    # Marker size
    ms = 3
    plotting_depths = ScalarField.from_expression(depths, "x").data * Xstar
    plt.plot(plotting_depths, sol.data[0], "v", ms = ms, label = "CA")
    plt.plot(plotting_depths, sol.data[1], "^", ms = ms, label = "CC")
    plt.plot(plotting_depths, sol.data[2], ">", ms = ms, label = "cCa")
    plt.plot(plotting_depths, sol.data[3], "<", ms = ms, label = "cCO3")
    plt.plot(plotting_depths, sol.data[4], "o", ms = ms, label = "Phi")
    plt.xlabel("Depth (cm)")
    plt.ylabel("Compositions and concentrations (dimensionless)")
    plt.legend(loc='upper right')
    plt.plot()
    plt.show()

if __name__ == '__main__':
    solution, covered_time, depths, Xstar = \
        integrate_equations(**asdict(Map_Scenario()))
    Plot_results(solution, covered_time, depths, Xstar)