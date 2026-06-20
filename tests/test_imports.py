"""Smoke tests: every module in the repository must import cleanly.

These are cheap but valuable -- they catch breakage from upstream API
changes (e.g. a removed ``scipy`` import path) before any physics runs.
"""
import importlib

import pytest

MODULES = [
    "GiltTNR2D",
    "GiltTNR2D_setup",
    "GiltTNR2D_test",
    "GiltTNR2D_envspec",
    "GiltTNR3D",
    "GiltTNR3D_setup",
    "GiltTNR3D_test",
    "GiltTNR3D_envspec",
    "GiltTNR3D_impurity",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    assert importlib.import_module(module_name) is not None
