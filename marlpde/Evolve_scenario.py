#!/usr/bin/env python

from dataclasses import asdict
import inspect
import os
import time
from datetime import datetime
import h5py
from LHeureux_model import LMAHeureuxPorosityDiff
from parameters import Map_Scenario, Solver, Tracker
from pde import CartesianGrid, ScalarField
from pde.grids.operators.cartesian import _make_derivative
from scipy.integrate import solve_ivp
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("AGG")

def integrate_equations(solver_parms, tracker_parms, pde_parms):
    '''Perform the integration and display and store the results.

    This function retrieves the parameters of the Scenario to be simulated and 
    the solution parameters for the integration. It then integrates the five
    partial differential equations from L'Heureux, stores and returns the 
    solution, to be used for plotting. Its inputs come from the parameters
    module, which has dataclasses governing the solver, storage and model specs,
    through the solver_parms, tracker_parms and pde_parms dicts, respectively.

    A progress bar shows how long the (remaining) integration will take.

    Parameters:
    -----------
    solver_parms: dict 
        Parameters about solver settings.
    tracker_parms: dict 
        Parameters about the progress bar and time interval for storage
    pde_parms: dict
        Model parameters, which govern e.g. the physical processes, but
        also the discretization, such as the number of grid cells.

    Returns:
    --------
    field_solutions: ndarray
        This is the "y" attribute of "sol" i.e. the solution derived by
        solve_ivp. See the scipy.integrate.solve_ivp documentation for
        some background. A reshape has been applied to arrive at one row per
        field. The solutions as a function of time have been removed such that
        only the solution for the last time is returned.

    covered_time: float
        This is the time interval of integration in years, from the start time
        (probably 0) until the requested final time. When the integration halted
        unexpectedly, the covered_time corresponds to the time covered until the
        integration halted.

    depths: pde.CartesianGrid
        These are the centers of the grid cells that together constitute the 
        grid.
    
    Xstar: float
        Scaling factor between physical depths and dimensionless depths as used
        in the differential equations.

    Store_folder: str
        Could be a relative path, i.e.  a path relative to the root of this 
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
    
    # I need those two fields for computing coA, which is rather involved.
    # There may be a simpler way of selecting these depths, but I haven't
    # found one yet. For now these two Heaviside step functions.
    not_too_shallow = ScalarField.from_expression(depths,
                      f"heaviside(x-{ShallowLimit/Xstar}, 0)")
    not_too_deep = ScalarField.from_expression(depths,
                   f"heaviside({DeepLimit/Xstar}-x, 0)")    
    
    # Not all keys from pde_parms are LMAHeureuxPorosityDiff arguments.
    # Taken from https://stackoverflow.com/questions/334655/passing-a-\
    # dictionary-to-a-function-as-keyword-parameters
    filtered_pde_parms = {k: v for k, v in pde_parms.items() if k in [p.name for
                          p in
                          inspect.signature(LMAHeureuxPorosityDiff).parameters.\
                          values()]}

    slices_all_fields = [slice(i * Number_of_depths, (i+1) * Number_of_depths) \
                             for i in range(5)]            

    eq = LMAHeureuxPorosityDiff(depths, slices_all_fields, not_too_shallow, 
                                not_too_deep, **filtered_pde_parms)     
    
    # Setting initial values for all five fields in three steps.
    # This is perhaps more involved than necessary, since the
    # end result is a 1D Numpy array with a length of five times
    # the number of depths. The number five comes from the five
    # fields we are tracking, by integrating five pdes.
    # First step.
    AragoniteSurface = ScalarField(depths, CAIni)
    CalciteSurface = ScalarField(depths, CCIni)
    CaSurface = ScalarField(depths, cCaIni)
    CO3Surface = ScalarField(depths, cCO3Ini)
    PorSurface = ScalarField(depths, PhiIni)
    
    # Second step.
    state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                         CO3Surface, PorSurface)
    # Third step.
    y0 = state.data.ravel()   
    
    no_progress_updates = tracker_parms["no_progress_updates"]

    start_computing = time.time()
    with tqdm(total=no_progress_updates) as pbar:
        t0 = solver_parms["t_span"][0]
        end_time = solver_parms["t_span"][1]
        progress_bar_args = [pbar, (end_time - t0) / no_progress_updates, t0]

        # The backend parameter determines which function should be called to 
        # determine the right-hand sides of the five pdes. It can be either
        # a Numpy-based function or a Numba-based function. The latter is 
        # faster.
        backend = solver_parms["backend"]
        # "backend" is not an argument for solve_ivp, so remove it now.
        del solver_parms["backend"]

        sol = solve_ivp(eq.fun if backend=="numpy" else eq.fun_numba, 
                        y0=y0, **solver_parms,
                        t_eval= tracker_parms["t_eval"], 
                        events = [eq.zeros, eq.zeros_CA, eq.zeros_CC, \
                        eq.ones_CA_plus_CC, eq.ones_Phi, eq.zeros_U, \
                        eq.zeros_W], args=progress_bar_args)
    end_computing = time.time()
    
    print()
    print(f"Number of rhs evaluations = {sol.nfev} \n")
    print(f"Number of Jacobian evaluations = {sol.njev} \n")
    print(f"Number of LU decompositions = {sol.nlu} \n")
    print(f"Status = {sol.status} \n")
    print(f"Success = {sol.success} \n")
    f = sol.t_events[0]
    print(("Times, in years, at which any field at any depth crossed zero: "\
      +', '.join(['%.2f']*len(f))+"") % tuple([Tstar * time for time in f]))
    print()
    g = sol.t_events[1]
    print(("Times, in years, at which CA at any depth crossed zero: "\
      +', '.join(['%.2f']*len(g))+"") % tuple([Tstar * time for time in g]))
    print()
    h = sol.t_events[2]
    print(("Times, in years, at which CC at any depth crossed zero: "\
      +', '.join(['%.2f']*len(h))+"") % tuple([Tstar * time for time in h]))
    print()
    k = sol.t_events[3]
    print(("Times, in years, at which CA + CC at any depth crossed one: "\
      +', '.join(['%.2f']*len(k))+"") % tuple([Tstar * time for time in k]))
    print()
    l = sol.t_events[4]
    print(("Times, in years, at which the porosity at any depth crossed one: "\
      +', '.join(['%.2f']*len(l))+"") % tuple([Tstar * time for time in l]))
    print()
    m = sol.t_events[5]
    print(("Times, in years, at which U at any depth crossed zero: "\
      +', '.join(['%.2f']*len(m))+"") % tuple([Tstar * time for time in m]))
    print()
    n = sol.t_events[6]
    print(("Times, in years, at which W at any depth crossed zero: "\
      +', '.join(['%.2f']*len(n))+"") % tuple([Tstar * time for time in n]))
    print()
    
    print(f"Message from solve_ivp = {sol.message} \n")
    print(("Time taken for solve_ivp is "
          f"{end_computing - start_computing:.2e}s. \n"))
    
    if sol.status == 0:
        covered_time = Tstar * (end_time - t0)
    else:
        covered_time = pbar.n * Tstar * (end_time - t0) / no_progress_updates 

    # Store your results somewhere in a subdirectory of a parent directory.
    store_folder = "../Results/" + \
                   datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
    os.makedirs(store_folder)
    stored_results = store_folder + "LMAHeureuxPorosityDiff.hdf5"
    # Keep a record of all parameters.
    stored_parms = solver_parms | tracker_parms | pde_parms
    # Remove items which will raise a problem when storing as metadata in an
    # hdf5 file
    if "jac_sparsity" in stored_parms:
        del stored_parms["jac_sparsity"]

    # Reshape the solutions such that you get one row per field.
    # The third axis, i.e.the time axis, can remain unchanged.
    field_solutions = sol.y.reshape(5, Number_of_depths, sol.y.shape[-1])

    with h5py.File(stored_results, "w") as stored:
        stored.create_dataset("solutions", data=field_solutions)
        stored.create_dataset("times", data=sol.t)
        for event_index, _ in enumerate(sol.t_events):
            stored.create_dataset("event_" + str(event_index), 
                                  data=sol.t_events[event_index])
        stored.attrs.update(stored_parms)

    # We will be plotting only the distributions corresponding to the last time.
    # Thus no point in returning all data. Moreover, all data have been saved.

    return field_solutions[:, :, -1], covered_time, depths, Xstar, store_folder

def Plot_results(last_field_sol, covered_time, depths, Xstar, store_folder):
    '''
    Plot the five fields at the end of the integration interval as a function
    of depth.
    '''
    fig, ax = plt.subplots()
    fig.suptitle(f"Distributions after {covered_time:.2e} years")
    # Marker size
    ms = 5
    plotting_depths = ScalarField.from_expression(depths, "x").data * Xstar

    ax.plot(plotting_depths, last_field_sol[0], "v", ms = ms, label = "CA")
    ax.plot(plotting_depths, last_field_sol[1], "^", ms = ms, label = "CC")
    ax.plot(plotting_depths, last_field_sol[2], ">", ms = ms, label = "cCa")
    ax.plot(plotting_depths, last_field_sol[3], "<", ms = ms, label = "cCO3")
    ax.plot(plotting_depths, last_field_sol[4], "o", ms = ms, label = "Phi")

    ax.set_xlabel("Depth (cm)")
    ax.set_ylabel("Compositions and concentrations (dimensionless)")
    ax.legend(loc='upper right')
    fig.savefig(store_folder + 'Final_distributions.pdf', bbox_inches="tight")

if __name__ == '__main__':
    pde_parms = asdict(Map_Scenario()) 
    solver_parms = asdict(Solver()) 
    tracker_parms = asdict(Tracker())
    Plot_results(*integrate_equations(solver_parms, tracker_parms, pde_parms))