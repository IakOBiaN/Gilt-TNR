"""Shared helpers for the Gilt-TNR test suite.

These functions mirror the physics computations that the ``*_test.py``
driver scripts perform, but without going through ``tntools.datadispenser``
(no disk caching), so the tests exercise this repository's algorithm code
directly and deterministically.
"""
import numpy as np
from ncon import ncon

from tntools import initialtensors, initialtensors_setup
import GiltTNR2D
import GiltTNR3D


# A complete set of parameters covering both the initial-tensor builders
# (tntools.initialtensors) and the Gilt-TNR algorithms. Individual tests
# override only what they care about via ``make_pars(**overrides)``.
_BASE_PARS = {
    # Model / initial tensor.
    "model": "ising",
    "beta": 0.4,
    "symmetry_tensors": True,
    "dtype": np.float64,
    "J": 1.0,
    "H": 0.0,
    # Initial-tensor pre-blocking flags (only relevant for some models).
    "initial2z": False,
    "initial2x2": False,
    "initial2x2x2": False,
    "initial4x4": False,
    # Coarse-graining (shared by 2D and 3D).
    "cg_eps": 1e-5,
    # Gilt, 2D.
    "gilt_eps": 1e-7,
    # Gilt, 3D.
    "gilt_eps_squares": 1e-4,
    "gilt_eps_cubes": 1e-4,
    "gilt_hastyquit": True,
    "gilt_split": True,
    "gilt_split_dynamic": True,
    "gilt_split_dynamic_eps": 1e-8,
    "gilt_split_dynamic_max_factor": 2.0,
    "gilt_split_factor": 1.0,
    # Shared.
    "gilt_print_envspec": False,
    "gilt_print_envspec_recursive": False,
    "verbosity": 0,
}


def make_pars(**overrides):
    """Return a fresh, complete parameter dict, with ``overrides`` applied."""
    pars = dict(_BASE_PARS)
    pars["cg_chis"] = range(1, 21)
    pars.update(overrides)
    return pars


# - 2D -

def run_2d(pars, n_iters):
    """Build the initial 2D tensor and apply ``n_iters`` Gilt-TNR steps."""
    A = initialtensors.get_initial_tensor(pars)
    log_fact = 0.0
    for _ in range(n_iters):
        A, log_fact = GiltTNR2D.gilttnr_step(A, log_fact, pars)
    return A, log_fact


def free_energy_2d(A, log_fact, beta, iter_count):
    """Free energy per site from a coarse-grained 2D tensor.

    Mirrors ``GiltTNR2D_test.get_free_energy``.
    """
    Z = ncon(A, [1, 2, 1, 2]).value()
    log_Z = np.log(Z) + log_fact
    F = -log_Z / beta
    return F / (2 * 4 ** iter_count)


def scaldims_2d(A, how_many):
    """Scaling dimensions from the transfer matrix of a 2D tensor.

    Mirrors ``GiltTNR2D_test.get_scaldims``.
    """
    transmat = ncon((A, A), [[3, -101, 4, -1], [4, -102, 3, -2]])
    es = transmat.eig([0, 1], [2, 3], hermitian=False)[0].to_ndarray()
    es = np.abs(es)
    es = -np.sort(-es)
    es[es == 0] += 1e-16
    log_es = np.log(es)
    log_es -= np.max(log_es)
    log_es /= -np.pi
    return log_es[:how_many]


# - 3D -

def run_3d_step(pars):
    """Build the initial 3D tensors and apply a single Gilt-TNR step.

    Returns ``(As, log_facts)`` after one step.
    """
    A, _ = initialtensors_setup.generate_A(pars=pars)
    As = (A,) * 8
    log_facts = [0] * 8
    out = GiltTNR3D.gilttnr_step(As, log_facts, pars)
    return out[0], out[1]


def free_energy_3d(As, log_facts, beta, iter_count):
    """Free energy per site from coarse-grained 3D tensors.

    Mirrors ``GiltTNR3D_test.get_free_energy``.
    """
    Z = ncon(
        (As[0], As[4], As[1], As[5]),
        ([1, 100, 3, 100, 4, 5], [3, 102, 1, 102, 10, 11],
         [6, 101, 8, 101, 5, 4], [8, 103, 6, 103, 11, 10]),
        order=([100, 101, 102, 103, 10, 11, 6, 8, 1, 3, 4, 5]),
    )
    log_fact = log_facts[0] + log_facts[1] + log_facts[4] + log_facts[5]
    logZ = np.abs(np.log(Z.value()) + log_fact)
    F = -logZ / beta
    return F / (8 ** (iter_count + 1))
