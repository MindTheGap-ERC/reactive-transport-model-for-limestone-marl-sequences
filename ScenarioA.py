## Parameters for Scenario A
#Taken from table 1 (p. 7)
from tracemalloc import start
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from LMAHeureuxPorosityDiffV2 import LMAHeureuxPorosityDiff
from pde import CartesianGrid, ScalarField
from scipy.integrate import solve_ivp
import time
from tqdm import tqdm
import os

Scenario = 'A'
CA0 = 0.6
CAIni = CA0
CC0 = 0.3
CCIni = CC0
KA = 10 ** (- 6.19)
KC = 10 ** (- 6.37)
cCa0 = 0.326e-3/np.sqrt(KC)
cCaIni = cCa0
cCO30 = 0.326e-3/np.sqrt(KC)
cCO3Ini = cCO30
Phi0 = 0.6
PhiIni = 0.5

ShallowLimit = 50

DeepLimit = 150

sedimentationrate = 0.1
m1 = 2.48
m2 = m1
n1 = 2.8
n2 = n1
rhos0 = 2.95 * CA0 + 2.71 * CC0 + 2.8 * (1 - (CA0 + CC0))

rhos = rhos0

rhow = 1.023
beta = 0.1
D0Ca = 131.9
k1 = 1

k2 = k1
k3 = 0.1
k4 = k3
muA = 100.09
DCa = 131.9
DCO3 = 272.6
b = 5e-4

PhiNR = Phi0

PhiInfty = 0.01

Xstar = D0Ca / sedimentationrate
Tstar = Xstar / sedimentationrate 

number_of_depths = 200

max_depth = 500

Depths = CartesianGrid([[0, max_depth * (1 + 0.5/number_of_depths)/Xstar]],\
                        [number_of_depths], periodic=False)

AragoniteSurface = ScalarField(Depths, CAIni)
CalciteSurface = ScalarField(Depths, CCIni)
CaSurface = ScalarField(Depths, cCaIni)
CO3Surface = ScalarField(Depths, cCO3Ini)
PorSurface = ScalarField(Depths, PhiIni) 

# I need those two fields for computing coA, which is rather involved.
# There may be a simpler way of selecting these depths, but I haven't
# found one yet. For now these two Heaviside step functions.
not_too_shallow = ScalarField.from_expression(Depths, 
                  "heaviside(x-{}, 0)".format(ShallowLimit/Xstar))
not_too_deep = ScalarField.from_expression(Depths, 
               "heaviside({}-x, 0)".format(DeepLimit/Xstar)) 

slices_for_all_fields = [slice(i * number_of_depths, (i+1) * number_of_depths) \
                         for i in range(5)]            

eq = LMAHeureuxPorosityDiff(Depths, slices_for_all_fields, CA0, CC0, cCa0, cCO30, Phi0, 
                            sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, 
                            m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, KA, KC, 
                            muA, D0Ca, PhiNR, PhiInfty, DCa, DCO3, 
                            not_too_shallow, not_too_deep)     

timeMult = 1
end_time = timeMult * Tstar/Tstar
# Number of times to evaluate, for storage.
no_t_eval = 100
t_eval = np.linspace(0, end_time, num = no_t_eval)

state = eq.get_state(AragoniteSurface, CalciteSurface, CaSurface, 
                     CO3Surface, PorSurface)

y0 = state.data.ravel()   

number_of_progress_updates = 100000

start_computing = time.time()
with tqdm(total=number_of_progress_updates, unit="‰") as pbar:
    sol = solve_ivp(fun = eq.fun_numba, t_span = (0, end_time), y0 = y0, \
                atol = 1e-3, rtol = 1e-3, t_eval= t_eval, \
                events = [eq.zeros, eq.zeros_CA, eq.zeros_CC, \
                eq.ones_CA_plus_CC, eq.ones_Phi, eq.zeros_U, eq.zeros_W],  \
                method="RK23", dense_output= True,\
                first_step = end_time/1e6, jac = eq.jac, \
                args=[pbar, [0, 1/number_of_progress_updates]])
end_computing = time.time()

print()
print("Number of rhs evaluations = {0}".format(sol.nfev))
print()
print("Number of Jacobian evaluations = {0}".format(sol.njev))
print()
print("Number of LU decompositions = {0}".format(sol.nlu))
print()
print("Status = {0}".format(sol.status))
print()
print("Success = {0}".format(sol.success))
print()
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

print("Message from solve_ivp = {0}".format(sol.message))
print()
print("Time taken for solve_ivp is {0:.2f}s.".format(end_computing - start_computing))
print()

if sol.status == 0:
    covered_time = Tstar * end_time
else:
   covered_time = pbar.n * Tstar/number_of_progress_updates 

plt.title("Situation after " + " {:.2f} ".format(covered_time) + " years")
# Marker size
ms = 3
depths = ScalarField.from_expression(Depths, "x").data * Xstar
plt.plot(depths, (sol.y)[slices_for_all_fields[0], -1], "v", ms = ms, label = "CA")
plt.plot(depths, (sol.y)[slices_for_all_fields[1], -1], "^", ms = ms, label = "CC")
plt.plot(depths, (sol.y)[slices_for_all_fields[2], -1], ">", ms = ms, label = "cCa")
plt.plot(depths, (sol.y)[slices_for_all_fields[3], -1], "<", ms = ms, label = "cCO3")
plt.plot(depths, (sol.y)[slices_for_all_fields[4], -1], "o", ms = ms, label = "Phi")
plt.xlabel("Depth (cm)")
plt.ylabel("Compositions and concentrations (dimensionless)")
plt.legend(loc='upper right')

# Copied from other branches
# Store your results somewhere in a subdirectory of a parent directory.
store_folder = "../Results/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S" + "/")
os.makedirs(store_folder)
stored_results = store_folder + "LMAHeureuxPorosityDiff.npy"
# Reshape the results into a more handy format
np.save(stored_results, np.array([(sol.y)[slices_for_all_fields[0], :] for i in range(5)]))
plt.tight_layout()
plt.savefig("../Results/Final_compositions_and_concentrations_" + datetime.now().\
                      strftime("%d_%m_%Y_%H_%M_%S") + ".png")
