# Tests

A regression test suite for Gilt-TNR, so the algorithm code can be changed
with confidence.

## Running

From the repository root:

```
pip install -r requirements-dev.txt
pytest                 # run everything (~25 s)
pytest -m "not slow"   # fast subset only (~5 s): imports, pure helpers, high-T 2D
```

## What is covered

| File | What it checks |
|------|----------------|
| `test_imports.py` | Every module imports cleanly (catches upstream API breakage). |
| `test_helpers.py` | Pure helper functions (`update_pars`, permutation utilities). |
| `test_gilttnr2d.py` | **Physical correctness** of the 2D algorithm: Ising free energy vs Onsager's exact solution (high-T and critical), and the Ising CFT scaling dimensions at criticality. Both the symmetric and dense tensor code paths. |
| `test_gilttnr3d.py` | **Characterization** of the 3D algorithm: output structure and a golden free-energy value after one step. The 3D code is still under development, so these pin current behaviour rather than asserting exactness. |

The tests build the initial tensors and call the Gilt-TNR steps directly
(via `tests/helpers.py`), bypassing `tntools.datadispenser`, so they do not
touch the on-disk `data/` cache and are fully deterministic.

## Updating the 3D golden value

If you intentionally change the 3D algorithm, the characterization test in
`test_gilttnr3d.py` will fail. Re-record `GOLDEN_FREE_ENERGY` from the new
output once you have confirmed the change is correct.
