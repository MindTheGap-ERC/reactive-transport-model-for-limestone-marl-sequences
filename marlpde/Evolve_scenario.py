#!/usr/bin/env python

from datetime import datetime
import os
from dataclasses import asdict
import inspect
import matplotlib.pyplot as plt
import h5py
from pde import CartesianGrid, ScalarField, FileStorage, LivePlotTracker
from pde import DataTracker
from pde.grids.operators.cartesian import _make_derivative
from parameters import Map_Scenario, Solver, Tracker
from LHeureux_model import LMAHeureuxPorosityDiff

def integrate_equations(solver_parms, tracker_parms, pde_parms):
    '''Perform the integration and display and store the results.

    This function uses the solver parameters for the integration and the
    parameters of the Scenario to be simulated as input.
    It integrates the five partial differential equations from L'Heureux, 
    stores and returns the solution, to be used for plotting. 
    Its inputs come from the parameters module, which has dataclasses governing
    the solver, storage and model specs, through the solver_parms, tracker_parms
    and pde_parms dicts, respectively.

    A progress bar shows how long the (remaining) integration will take.

    Parameters:
    -----------
    
    - **solver_parms** (*dict*):  
      Parameters about solver settings.
    
    - **tracker_parms** (*dict*):  
      Parameters about the progress bar and time interval for storage.
    
    - **pde_parms** (*dict*):  
      Model parameters, which govern e.g. the physical processes, but 
      also the discretization, such as the number of grid cells.

    Returns:
    --------

    - **sol** (*pde.FieldCollection(pde.ScalarField(..), ..*))
         The data attribute of a pde.FieldCollection contains the final 
         solutions of the integration as a Numpy ndarray of floats.
         sol.data[0] through sol.data[4] contain the final solutions for the
         calcite and aragonite compositions, calcium ions and carbonate ions
         concentrations and the porosity, respectively.

    - **covered_time** (*float*)
        This is the time interval of integration in years. When the integration
        halted unexpectedly, the covered_time corresponds to the time covered 
        until the integration halted.

    - **depths** (*pde.CartesianGrid*)
        These are the centers of the grid cells that together constitute the 
        grid.
    
    - **Xstar** (*float*)
        Scaling factor between physical depths and dimensionless depths as used
        in the differential equations.

    - **Store_folder** (*str*)
        Could be a relative path, i.e. a path relative to the root of this 
        repository, to a folder where solutions of the integration should be 
        stored. Could also be an absolute path, though.
    '''

    Xstar = pde_parms["Xstar"]
    Tstar = pde_parms["Tstar"]
    max_depth = pde_parms["max_depth"]
    ShallowLimit = pde_parms["ShallowLimit"]
    DeepLimit = pde_parms["DeepLimit"]
    CAIni = pde_parms["CAIni"]
    CCIni = pde_parms["CCIni"]
    cCaIni = pde_parms["cCaIni"]
    cCO3Ini = pde_parms["cCO3Ini"]
    PhiIni = pde_parms["PhiIni"]

    Number_of_depths = pde_parms["N"]

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
    filtered_kwargs = {k: v for k, v in pde_parms.items() if k in [p.name for p
                       in inspect.signature(LMAHeureuxPorosityDiff).parameters.\
                        values()]}

    eq = LMAHeureuxPorosityDiff(AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface, not_too_shallow, 
                                not_too_deep, **filtered_kwargs)             
    
    state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                         CO3Surface, PorSurface)
    
    # Store your results somewhere in a subdirectory of a parent directory.
    store_folder = "../Results/" + \
                   datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
    os.makedirs(store_folder)
    stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
    # Keep a record of all parameters.
    storage_parms = solver_parms | tracker_parms | pde_parms
    storage = FileStorage(stored_results, info=storage_parms)

    if tracker_parms["live_plotting"]:
        live_plots = LivePlotTracker(interval=\
                                     tracker_parms["plotting_interval"], \
                                     title="Integration results",
                                     show=True, max_fps=1, \
                                     plot_args ={"ax_style": {"ylim": (0, 1.5)}})
    else:
        live_plots = None

    if tracker_parms["track_U_at_bottom"]:
        data_tracker = DataTracker(eq.track_U_at_bottom, \
                               interval = tracker_parms["data_tracker_interval"])
    else:
        data_tracker = None

    sol, info = eq.solve(state, **solver_parms, tracker=["progress", \
                         storage.tracker(\
                             tracker_parms["progress_tracker_interval"]),\
                         live_plots, data_tracker])

    print()
    print(f"Meta-information about the solution : {info} \n")        

    covered_time_span = Tstar * info["controller"]["t_final"]

    if tracker_parms["track_U_at_bottom"]:
        with h5py.File(stored_results, 'a') as hf:
            U_grp = hf.create_group("U")
            U_grp.create_dataset("U_at_bottom", \
                              data=data_tracker.dataframe.to_numpy())

    return sol, covered_time_span, depths, Xstar, store_folder

def Plot_results(sol, covered_time, depths, Xstar, store_folder):
    '''
    Plot the five fields at the end of the integration interval as a function
    of depth.
    '''
    fig, ax = plt.subplots()
    fig.suptitle(f"Distributions after {covered_time:.2f} years")
    # Marker size
    ms = 5
    plotting_depths = ScalarField.from_expression(depths, "x").data * Xstar
    ax.plot(plotting_depths, sol.data[0], "v", ms = ms, label = "CA")
    ax.plot(plotting_depths, sol.data[1], "^", ms = ms, label = "CC")
    ax.plot(plotting_depths, sol.data[2], ">", ms = ms, label = "cCa")
    ax.plot(plotting_depths, sol.data[3], "<", ms = ms, label = "cCO3")
    ax.plot(plotting_depths, sol.data[4], "o", ms = ms, label = "Phi")
    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Compositions and concentrations (dimensionless)")
    ax.legend(loc='upper right')
    fig.savefig(store_folder + 'Final_distributions.pdf', bbox_inches="tight")

if __name__ == '__main__':
    pde_parms = asdict(Map_Scenario()) 
    solver_parms = asdict(Solver()) 
    tracker_parms = asdict(Tracker())
    solution, covered_time, depths, Xstar, store_folder = \
        integrate_equations(solver_parms, tracker_parms, pde_parms)
    Plot_results(solution, covered_time, depths, Xstar, store_folder)
