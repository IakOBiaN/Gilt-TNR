"""Pytest configuration and shared fixtures for the Gilt-TNR test suite."""
import os
import sys
import logging

import pytest

# Make both the repository root (for GiltTNR2D etc.) and this tests
# directory (for ``helpers``) importable regardless of how pytest is
# invoked.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The algorithm modules emit a lot of INFO logging; keep test output clean.
logging.disable(logging.WARNING)

from tntools import modeldata  # noqa: E402
import helpers  # noqa: E402


@pytest.fixture(scope="session")
def critical_beta():
    """Exact critical inverse temperature of the 2D Ising model."""
    return modeldata.get_critical_beta({"model": "ising", "H": 0.0, "J": 1.0})


@pytest.fixture(scope="session")
def cg_2d_critical(critical_beta):
    """A 2D Ising tensor coarse-grained to the critical fixed point.

    Computed once per session (it is the most expensive 2D run) and shared
    by the critical free-energy and scaling-dimension tests.

    Returns ``(A, log_fact, pars, n_iters)``.
    """
    n_iters = 5
    pars = helpers.make_pars(
        beta=critical_beta,
        gilt_eps=1e-6,
        cg_chis=range(1, 17),
        cg_eps=1e-5,
    )
    A, log_fact = helpers.run_2d(pars, n_iters)
    return A, log_fact, pars, n_iters
