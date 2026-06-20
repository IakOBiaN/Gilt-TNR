"""Physics regression tests for the 2D (square lattice) Gilt-TNR algorithm.

The square-lattice implementation is the mature one, so these assert real
physical correctness against known exact results for the 2D Ising model:
Onsager's free energy, and the Ising CFT scaling dimensions at criticality.
"""
import pytest
from tntools import modeldata

import helpers


def test_free_energy_high_temperature():
    """Free energy at beta=0.4 must match Onsager's exact solution."""
    pars = helpers.make_pars(beta=0.4, gilt_eps=1e-7,
                             cg_chis=range(1, 21), cg_eps=1e-5)
    n_iters = 6
    A, log_fact = helpers.run_2d(pars, n_iters)
    f = helpers.free_energy_2d(A, log_fact, pars["beta"], n_iters)
    exact = modeldata.get_free_energy(pars)
    assert abs(f - exact) / abs(exact) < 1e-5


def test_free_energy_without_symmetry_tensors():
    """The non-symmetric (dense) tensor code path must also be correct."""
    pars = helpers.make_pars(beta=0.4, symmetry_tensors=False,
                             gilt_eps=1e-7, cg_chis=range(1, 21), cg_eps=1e-5)
    n_iters = 5
    A, log_fact = helpers.run_2d(pars, n_iters)
    f = helpers.free_energy_2d(A, log_fact, pars["beta"], n_iters)
    exact = modeldata.get_free_energy(pars)
    assert abs(f - exact) / abs(exact) < 1e-4


@pytest.mark.slow
def test_free_energy_critical(cg_2d_critical):
    """Free energy at the critical point must match the exact solution.

    The shared critical fixture uses a modest bond dimension (chi=16) and
    only 5 iterations -- tuned to land on the critical fixed point for the
    scaling-dimension test -- so the free energy here converges to ~7e-4
    rather than the ~1e-6 reachable with larger chi and more iterations.
    """
    A, log_fact, pars, n_iters = cg_2d_critical
    f = helpers.free_energy_2d(A, log_fact, pars["beta"], n_iters)
    exact = modeldata.get_free_energy(pars)
    assert abs(f - exact) / abs(exact) < 2e-3


@pytest.mark.slow
def test_scaling_dimensions_critical(cg_2d_critical):
    """The leading scaling dimensions must match the 2D Ising CFT.

    Exact values: 0 (identity), 1/8 (spin), 1 (energy), 9/8, 9/8, 2, ...
    """
    A, _log_fact, _pars, _n = cg_2d_critical
    sd = helpers.scaldims_2d(A, 6)
    assert abs(sd[0] - 0.0) < 1e-6      # identity
    assert abs(sd[1] - 0.125) < 0.02    # spin sigma
    assert abs(sd[2] - 1.0) < 0.03      # energy epsilon
    assert abs(sd[3] - 1.125) < 0.05    # descendants of sigma
    assert abs(sd[4] - 1.125) < 0.05
    assert abs(sd[5] - 2.0) < 0.1
    # Scaling dimensions are returned in non-decreasing order.
    assert list(sd) == sorted(sd)
