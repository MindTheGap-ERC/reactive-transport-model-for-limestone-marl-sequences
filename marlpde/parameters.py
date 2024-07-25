import configparser
from pathlib import Path
from dataclasses import (dataclass, asdict, make_dataclass, fields)
from subprocess import (run)
import h5py as h5
from pint import UnitRegistry
import numpy as np

u = UnitRegistry()
quantity = u.Quantity

@dataclass
class Scenario:
    '''
    Sets all the Scenario parameter values from the FORTRAN code from 
    L'Heureux (2018). Strictly, the initial and boundary porosities are not
    part of the Scenario parameters, but they are included here.
    '''
    mua: quantity    = 100.09 * u.g/u.mol
    rhoa: quantity   = 2.95 * u.g/u.cm**3
    rhoc: quantity   = 2.71 * u.g/u.cm**3
    rhot: quantity   = 2.8 * u.g/u.cm**3
    rhow: quantity   = 1.023 * u.g/u.cm**3
    D0ca: quantity   = 131.9 * u.cm**2/u.a
    D0co3: quantity  = 272.6 * u.cm**2/u.a
    Ka: quantity     = 10**(-6.19) * u.M**2
    Kc: quantity     = 10**(-6.37) * u.M**2
    beta: quantity   = 0.1 * u.cm / u.a
    b: quantity      = 5.0 / u.kPa
    k1: quantity     = 1.0 / u.a
    k2: quantity     = 1.0 / u.a
    k3: quantity     = 0.1 / u.a
    k4: quantity     = 0.1 / u.a
    nn: quantity     = 2.8 * u.dimensionless
    m: quantity      = 2.48 * u.dimensionless
    S: quantity      = 0.1 * u.cm / u.a
    # cAthy: quantity  = 0.1 * u.dimensionless
    phiinf: quantity = 0.01 * u.dimensionless
    phi0: quantity   = 0.8 * u.dimensionless
    ca0: quantity    = 0.326e-3 * u.M
    co30: quantity   = 0.326e-3 * u.M
    ccal0: quantity  = 0.3 * u.dimensionless
    cara0: quantity  = 0.6 * u.dimensionless
    xdis: quantity   = 50.0 * u.cm       # x_d   (start of dissolution zone)
    length: quantity = 500.0 * u.cm
    Th: quantity     = 100.0 * u.cm      # h_d   (height of dissolution zone)
    phi00: quantity  = 0.8 * u.dimensionless
    ca00: quantity   = 0.326e-3 * u.M    # sqrt(Kc) / 2
    co300: quantity  = 0.326e-3 * u.M    # sqrt(Kc) / 2
    ccal00: quantity = 0.3 * u.dimensionless
    cara00: quantity = 0.6 * u.dimensionless

def Map_Scenario():
    '''
    Maps the Fortran 'Scenario' parameters to the equivalent parameters in the
    Matlab and Python codes. These codes use slightly different parameter 
    names. Besides the mapping, also a few numerical conversions are applied.
    To do these conversions, a number of FORTRAN parameters need to be passed on 
    "as is", i.e. unchanged.
    Units from Scenario are dropped, so only magnitudes (i.e. values) are 
    retained.
    '''
    mapping = {"Ka":"KA",
               "Kc":"KC", 
               "cara0": "CA0", 
               "cara00": "CAIni",
               "ccal0": "CC0",
               "ccal00": "CCIni",
               "ca0": "ca0",
               "ca00": "ca00",
               "co30": "co30",
               "co300": "co300",               
               "phi0": "Phi0", 
               "phi00":"PhiIni", 
               "xdis": "ShallowLimit",
               "Th": "Th",
               "S": "sedimentationrate",
               "m": "m1",
               "nn": "n1",
               "rhoa": "rhoa",
               "rhoc": "rhoc",
               "rhot": "rhot",
               "rhow": "rhow",
               "beta": "beta",
               "b": "b",
               "D0ca": "D0Ca",
               "k1": "k1",
               "k2": "k2",
               "k3": "k3",
               "k4": "k4",
               "mua": "muA",
               "D0co3": "DCO3",  
               "phiinf": "PhiInfty",
               "length": "max_depth"
    }

    all_fields = fields(Scenario)

    allowed_fields = [field for field in all_fields
                      if field.name in mapping]
    
    derived_fields = [(mapping[field.name], field.type, field.default.magnitude) 
                      for field in allowed_fields]
                    
    # Here we append the fields that cannot be initialised directly from the
    # Scenario dataclass.
    derived_fields.extend([("cCa0", float, None),
                           ("cCaIni", float, None),
                           ("cCO30", float, None),
                           ("cCO3Ini", float, None),
                           ("DeepLimit", float, None),
                           ("rhos0", float, None),
                           ("rhos", float, None),
                           ("Xstar", float, None),
                           ("Tstar", float, None),
                           ("m2", float, None),
                           ("n2", float, None),
                           ("DCa", float, None),
                           ("PhiNR", float, None)])

    def post_init(self):
        # The Python parameters that need additional conversion
        # are initialised here.   
        self.cCa0 = self.ca0/np.sqrt(self.KC)
        self.cCaIni = self.ca00/np.sqrt(self.KC)
        self.cCO30 =  self.co30/np.sqrt(self.KC)
        self.cCO3Ini = self.co300/np.sqrt(self.KC)
        self.DeepLimit = self.ShallowLimit + self.Th
        self.rhos0 = self.rhoa * self.CA0 + self.rhoc * self.CC0 + \
                     self. rhot * (1 - (self.CA0 + self.CC0))
        self.rhos = self.rhos0
        self.Xstar = self.D0Ca / self.sedimentationrate
        self.Tstar = self.Xstar / self.sedimentationrate
        self.b = self.b/1e4
        self.m2 = self.m1
        self.n2 = self.n1
        self.DCa = self.D0Ca
        self.PhiNR = self.PhiIni

               
    derived_dataclass = make_dataclass("Mapped parameters", derived_fields,
                                       namespace={"__post_init__": post_init})

    return derived_dataclass()

@dataclass
class Solver:
    '''
    Initialises all the parameters for the solver.
    So parameters like time interval, time step and tolerance.
    '''
    dt: float     = 1.e-6
    # tmax is the integration time in years.
    tmax: int     = Map_Scenario().Tstar
    N: int        = 200
    solver: str   = "scipy"
    scheme: str   = "euler"
    backend: str = "numba"
    retinfo: bool = True
    adaptive: bool = True

@dataclass
class Tracker:
    '''
    Initialises all the tracking parameters, such as tracker interval.
    Also indicates the quantities to be tracked, as boolean values.
    '''
    progress_tracker_interval: float = Solver().tmax/ 100 / Map_Scenario().Tstar
    live_plotting: bool = False
    plotting_interval: str = '0:05'
    data_tracker_interval: float = 0.01
    track_U_at_bottom: bool = False


