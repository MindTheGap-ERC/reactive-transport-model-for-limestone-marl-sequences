from dataclasses import (dataclass, make_dataclass, fields)
from pint import UnitRegistry
import numpy as np
from scipy.sparse import lil_matrix, dia_matrix, csr_matrix

u = UnitRegistry()
quantity = u.Quantity

def no_init_docs(cls):
    cls.__init__.__doc__ = "@private"
    return cls


@no_init_docs
@dataclass
class Scenario:
    """ Sets all the Scenario parameter values from the FORTRAN code from 
    L'Heureux (2018). Strictly, the initial and boundary porosities are not
    part of the Scenario parameters, but they are included here.
    """
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
                           ("PhiNR", float, None),
                           ("N", int, None)])

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
        self.b = (self.b/1e4) 
        self.m2 = self.m1
        self.n2 = self.n1
        self.DCa = self.D0Ca
        self.PhiNR = self.PhiIni
        self.N = 200

               
    derived_dataclass = make_dataclass("Mapped parameters", derived_fields,
                                       namespace={"__post_init__": post_init})

    return derived_dataclass()

def jacobian_sparsity():
    '''
    It turns out the the Jacobian sparsity matrix, as provided by
    previous versions of this function was too strict. The diagonals
    had a width of 1, this implicitly assumes no dependencies of the
    Jacobian elements on fields at adjacent depths. A run with such
    a Jacobian sparsity matrix took a lot of compute power and time and
    ultimately halted without covering a single time step! A debugging 
    session with breakpoints inside solve_ivp showed that the Jacobian 
    matrix computed by `solve_ivp` (when neither the Jacobian nor the 
    Jacobian sparsity matrix was provided) has non-zero elements not 
    only on the diagonals but also on adjacent positions along the 
    diagonals. This was confirmed by ChatGPT:
    "It is indeed common for the Jacobian matrix of coupled partial 
    differential equations (PDEs), such as advection-diffusion equations, 
    to have non-zero elements beyond the main diagonal. This phenomenon, 
    where adjacent elements in the Jacobian are non-zero, is often due 
    to the spatial discretization of the differential equations."
    Likewise computing a functional Jacobian is only feasible for the
    diagonals, for positions near the diagonals it is very hard.

    Here we will choose diagonals of width 3.
    '''

    no_depths = Map_Scenario().N
    # Number of fields = 5, i.e. calcite, aragonite, concentrations of
    # calcium and carbonate ions and the porosity.
    no_fields = 5
    n = no_fields * no_depths
    
    diagonals = np.ones((3 * (2 * no_fields - 1), n))

    # Construct the offsets.
    offsets = ()
    for offset in range(-n + no_depths, n - no_depths + 1, no_depths): 
        offsets = offsets + ((offset -1, offset, offset + 1),)
    # Turn offsets into a single tuple instead of a tuple of tuples.
    offsets = sum(offsets, ())
    # Check that we have as many offsets as diagonals:
    try:
        assert len(offsets) == diagonals.shape[0]
    except AssertionError as e:
        print('Setup of diagonals incorrect.')
    # Construct the sparse matrix. 
    raw_matrix = lil_matrix(dia_matrix((diagonals, offsets), shape=(n, n))) 
    # Set the Jacobian elements to zero that correspond with the derivatives
    # of the rhs of equations 40 and 41 wrt phi.
    raw_matrix[:2 * no_depths, 4 * no_depths:] = 0

    return csr_matrix(raw_matrix)


@no_init_docs
@dataclass
class Solver():
    """Initialises all the parameters for the solver, i.e. parameters like time
    interval, time step and tolerance.
    """
    dt: float = 1.e-6
    """ Initial time step for the integration, in units of T* (dimensionless).
    If the time step is not adaptive, i.e. if 'adaptive = False', this time 
    step will be applied throughout the integration. If None, a default value
    based on the stepper will be chosen."""
    t_range: float = 1
    """ The time interval that the integration should cover in units of T*
    (dimensionless)."""
    solver: str = "scipy"
    """The type of solver to use."""
    scheme: str = "euler"
    """The type of integration scheme to use. Beware that 'scheme'
    will only be propagated if you have chosen py-pde's native
    'explicit' solver as 'solver'."""
    adaptive: bool = True
    """Whether or not to use an adaptive time step. Beware that 'adaptive' 
    will only be propagated if you have chosen py-pde's native
    'explicit' solver as 'solver'."""
    method: str = "LSODA"
    """solve_ivp from scipy.integrate offers six methods. 
    They can be set here."""
    lband: int = 1
    """ Setting lband and uband for method="LSODA" leads to tremendous
    performance increase. See Scipy's 'solve_ivp' documentation for background.
    Consider it equivalent to providing a sparsity matrix for the "Radau" 
    and "BDF" implicit methods."""
    uband: int = 1
    """ Setting lband and uband for method="LSODA" leads to tremendous
    performance increase. See Scipy's 'solve_ivp' documentation for background.
    Consider it equivalent to providing a sparsity matrix for the "Radau" 
    and "BDF" implicit methods."""
    backend: str = "numba"
    """Options are 'numpy and 'numba'. Choosing 'numba' means that the 
    computations of the right-hand sides of the partial differential equations
    for the five fields that we are integrating are compiled. This option is 
    almost always faster than 'numpy', which uses the Python interpreter to 
    launch (compiled) Numpy routines. However, the 'numba' option comes with
    limitations with respect to the Python and Numpy routines that can be 
    deployed. See the Numba documentation for a list of supported Numpy 
    features.
    """
    ret_info: bool = True
    """Flag determining whether diagnostic information about the solver 
    process should be returned. """

    def __post_init__(self):
        '''
        Filter out solver settings that are mutually incompatible.
        '''
        try:
            if self.solver != "scipy":
                del self.__dataclass_fields__["method"]
                del self.__dataclass_fields__["lband"]
                del self.__dataclass_fields__["uband"]
            else:
                del self.__dataclass_fields__["scheme"]
                del self.__dataclass_fields__["adaptive"]
                if self.method != "LSODA":
                    del self.__dataclass_fields__["lband"]
                    del self.__dataclass_fields__["uband"]
        except KeyError:
            pass


@dataclass
class Tracker:
    '''
    Initialises all the tracking parameters, such as tracker interval.
    Also indicates the quantities to be tracked, as boolean values.
    '''
    progress_tracker_interval: float = Solver().t_range / 1_000
    """This determines how often the progress bar, showing the 
    integration progress, is updated. It is expressed in integration
    units, i.e. in units of T*."""
    live_plotting: bool = True
    """It is possible to see how the distributions over depth, of the five
    fields that we are integrating, evolve as integration progresses."""
    plotting_interval: str = '0:05'
    """Attribute of the LivePlotTracker class that determines the interruption
    of the simulation. A single number determines an interval (measured in the 
    simulation time unit) of regular interruption. A string is interpreted as a
    duration in real time assuming the format ‘hh:mm:ss’. A list of numbers is 
    taken as explicit simulation time points."""
    data_tracker_interval: float = 0.01
    """Attribute of the DataTracker class that determines the interruption
    of the simulation. A single number determines an interval (measured in the 
    simulation time unit) of regular interruption. A string is interpreted as a
    duration in real time assuming the format ‘hh:mm:ss’. A list of numbers is 
    taken as explicit simulation time points."""
    track_U_at_bottom: bool = False
    """Whether to track solid matrix advection velocity U at the bottom of the
    grid and store it as a dataset 'U at bottom' in a separate group 'U' 
    within the hdf5 file containing all stored results."""