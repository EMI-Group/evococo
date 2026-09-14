# Regression tests

From the repository root, with the project dependencies installed:

```bash
python -B -m unittest discover -s tests -v
node --test tests/test_frontend_regressions.cjs
```

The Python suite covers model failure propagation, WebSocket authentication,
benchmark cache identity and summaries, and migration prelude injection. Model
requests are mocked; the tests never contact an LLM provider. With PyTorch/EvoX
installed, it also runs small CPU subprocess trials to check full-population
fitness validation and a generated GDE3 implementation. These CPU tests are
explicitly skipped when their dependencies are missing.

To include the standalone MATLAB output-protocol test, make `matlab` available
on `PATH` and set `EVOCOCO_TEST_MATLAB=1` for the test process. For example, in
PowerShell:

```powershell
$env:EVOCOCO_TEST_MATLAB = '1'
python -B -m unittest discover -s tests -p test_evaluation_regressions.py -v
Remove-Item Env:EVOCOCO_TEST_MATLAB
```

The MATLAB test uses a separate batch process and temporary directories. It
does not require PlatEMO or start a parallel pool. Existing benchmark outputs
are not used or modified by these tests.
