"""Compatibility of solve_ivp output with the release HDF5 readers."""
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from pde import CartesianGrid, FieldCollection, FileStorage, ScalarField
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'marlpde'))
from Evolve_scenario import integrate_equations, store_solution
from LHeureux_model import LMAHeureuxPorosityDiff
from parameters import Map_Scenario, Tracker


def test_release_layout_and_bottom_velocity(tmp_path):
    grid = CartesianGrid([[0, 1]], [3])
    state = FieldCollection([ScalarField(grid, 0.) for _ in range(5)])
    data = np.arange(4*5*3, dtype=float).reshape(4, 5, 3) / 100
    data[:, 4, :] = np.array([[.6, .7, .8], [.7, .6, .5],
                              [.4, .5, .6], [.5, .4, .3]])
    equation = object.__new__(LMAHeureuxPorosityDiff)
    equation.Phi_sl = slice(12, 15)
    equation.presum, equation.rhorat = -2., 1.5
    times = np.array([0., .2, .4, .6])
    sol = SimpleNamespace(t=times, y=data.reshape(4, 15).T,
                          t_events=[np.array([.3]), np.array([])],
                          success=True, message='Reached final time')
    filename = tmp_path / 'output.hdf5'
    store_solution(filename, sol, state, equation,
                   {'Tstar': 13190., 'backend': 'numba', 't_eval': times})
    with h5py.File(filename) as f:
        assert_array_equal(f['data'], data)
        assert_array_equal(f['times'], times)
        assert_array_equal(f['U/U_at_bottom'][:, 0], times)
        phi = data[:, 4, -1]
        expected = -2. + 1.5 * phi**3 * (1-np.exp(10-10/phi))/(1-phi)
        assert_allclose(f['U/U_at_bottom'][:, 1], expected)
        assert float(f.attrs['Tstar']) == 13190.
        assert_array_equal(f['event_0'], [.3])
        assert f['event_1'].shape == (0,)
    # The release uses py-pde FileStorage, which requires field metadata too.
    storage = FileStorage(filename)
    assert_array_equal(storage[2].data, data[2])
    assert_array_equal(storage.times, times)
    storage.close()
    assert_array_equal(state.data, np.zeros((5, 3)))


@pytest.mark.parametrize('backend', ['numpy', 'numba'])
def test_short_integration(tmp_path, monkeypatch, backend):
    run = tmp_path / 'run'
    run.mkdir()
    monkeypatch.chdir(run)
    parameters = asdict(Map_Scenario())
    parameters['N'] = 8
    solver = dict(backend=backend, method='Radau', t_span=(0., 1e-4),
                  first_step=1e-6, rtol=1e-3, atol=1e-3, dense_output=False)
    tracker = dict(no_progress_updates=10, t_eval=np.linspace(0, 1e-4, 4))
    final, covered, _, _, folder = integrate_equations(solver, tracker, parameters)
    assert solver['backend'] == backend
    with h5py.File(Path(folder) / 'LMAHeureuxPorosityDiff.hdf5') as f:
        assert f['data'].shape == (4, 5, 8)
        assert f['U/U_at_bottom'].shape == (4, 2)
        assert_array_equal(f['data'][-1], final)
        assert_array_equal(f['times'], tracker['t_eval'])
        assert np.all(np.isfinite(f['U/U_at_bottom']))
        assert_allclose(f['U/U_at_bottom'][0, 1], 1.)
        assert float(f.attrs['Tstar']) == parameters['Tstar']
    assert_allclose(covered, parameters['Tstar'] * 1e-4)


def test_default_snapshot_count():
    tracker = Tracker()
    assert len(tracker.t_eval) == 1001
