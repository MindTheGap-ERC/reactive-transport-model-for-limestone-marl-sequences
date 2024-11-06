"""
.. include:: ../README.md
"""
import pathlib
import sys
from .LHeureux_model import LMAHeureuxPorosityDiff
from .parameters import Map_Scenario, Solver, Tracker
from .parameters import Scenario, jacobian_sparsity

# Control what pdoc will document, i.e. output to .html.
__all__ = ['LMAHeureuxPorosityDiff', 'Scenario', 'Map_Scenario', 
           'jacobian_sparsity', 'Solver', 'Tracker']

sys.path.append(str(pathlib.Path(__file__).parent))
