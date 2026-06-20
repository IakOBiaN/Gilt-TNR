"""Characterization tests for the 3D (cubical lattice) Gilt-TNR algorithm.

The 3D implementation is still under development (per the README: bugs are
possible and the design may change), and the 3D Ising model has no simple
closed-form free energy. So rather than asserting physical exactness, these
tests pin down the *current* behaviour: structure of the output and a golden
free-energy value. They guard against accidental changes during refactoring;
if the algorithm is intentionally changed, update GOLDEN_FREE_ENERGY.
"""
import pytest

import helpers

# Free energy per site after one Gilt-TNR step from the initial tensor,
# for the ising3d model at beta=0.22165 with the settings below.
# Recorded on 2026-06-21 (numpy 1.26.4, this repo's code).
GOLDEN_FREE_ENERGY = -3.5528792048


@pytest.fixture(scope="module")
def one_step_3d():
    pars = helpers.make_pars(
        model="ising3d",
        beta=0.22165,
        initial2z=True,
        cg_chis=range(1, 10),
        cg_eps=1e-5,
        gilt_eps_squares=1e-4,
        gilt_eps_cubes=1e-4,
        gilt_hastyquit=True,
    )
    As, log_facts = helpers.run_3d_step(pars)
    return As, log_facts, pars


@pytest.mark.slow
def test_3d_step_output_structure(one_step_3d):
    As, log_facts, _pars = one_step_3d
    assert len(As) == 8
    # Every coarse-grained tensor is rank 6 (a cubical-lattice tensor).
    for A in As:
        assert len(A.shape) == 6


@pytest.mark.slow
def test_3d_free_energy_characterization(one_step_3d):
    As, log_facts, pars = one_step_3d
    f = helpers.free_energy_3d(As, log_facts, pars["beta"], 1)
    assert f == pytest.approx(GOLDEN_FREE_ENERGY, rel=1e-3)
