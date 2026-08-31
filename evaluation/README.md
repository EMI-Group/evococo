# Evaluation

The scripts in this directory evaluate generated EvoX algorithms without modifying the generated
files. Run all commands from the repository root. Public entry points start with `run_`; files that
start with `_` are isolated workers and should not normally be invoked directly.

## Migration reliability

Check syntax, Ruff diagnostics, execution, and DTLZ2 convergence:

```bash
python evaluation/run_migration_reliability_benchmark.py \
  --dir experiments/generated_algorithms \
  --workers 1
```

Use one worker when GPU memory is limited. The report is written to
`experiments/generated_algorithms/benchmark_report.json`.

## Optimization fidelity

`run_optimization_fidelity_benchmark.py` runs independent optimization trials and summarizes the
resulting IGD distribution. Select a benchmark suite with `--suite`:

| Suite | Problems | Default dimensions for 3 objectives | Implementation |
|---|---|---|---|
| `DTLZ` | DTLZ1–DTLZ7 | 7, 12, or 22 | EvoMO |
| `WFG` | WFG1–WFG9 | 12 | EvoMO |
| `LSMOP` | LSMOP1–LSMOP9 | 300 | EvoMO |
| `MaF` | MaF1–MaF15 | 2–60 | EvoMO |

Use `--suite all` for all 40 problems. `--problems` optionally selects a subset belonging to the
chosen suite.

Run the default protocol—21 seeds per algorithm/problem pair—on physical GPU 0:

```bash
python evaluation/run_optimization_fidelity_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --suite DTLZ \
  --gpu 0
```

The full command is large: 48 algorithms × 7 problems × 21 runs is 7,056 isolated trials. Validate
new code with a small run first:

```bash
python evaluation/run_optimization_fidelity_benchmark.py \
  --algorithm-file experiments/generated_algorithms/AGE-MOEA.py \
  --suite DTLZ \
  --problems DTLZ2 \
  --runs 2 \
  --generations 10 \
  --gpu 0 \
  --output-dir evaluation_results/fidelity_smoke
```

The output directory contains:

- `trials.csv`: one row per seed, including IGD, runtime, failures, OOMs, and timeouts.
- `summary.csv`: coverage, mean/median/standard-deviation IGD, and mean runtime.

Successful trials are resumable. Failed and timed-out trials are retained in `trials.csv` and
retried on the next invocation.

### Optional PlatEMO comparison

The repository includes `run_platemo_optimization_fidelity_benchmark.m` for generating the
reference results with PlatEMO. Set `PLATEMO_ROOT` or add PlatEMO to the MATLAB path, then run:

```matlab
run('evaluation/run_platemo_optimization_fidelity_benchmark.m')
```

It runs the same 48 algorithm names on DTLZ, WFG, LSMOP, and MaF over 21 seeds and writes
`evaluation_results/platemo_fidelity/platemo_reference.csv`. Reference rows include `ValidRuns`
and `ExpectedRuns`; the Python evaluator only uses complete PlatEMO references. Per-generation
objectives and IGD are written as one JSON object per line under `per_generation_jsonl/`.

Pass a reference summary to classify the generated mean IGD:

```bash
python evaluation/run_optimization_fidelity_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --problems DTLZ2 \
  --reference-csv path/to/platemo_reference.csv \
  --gpu 0
```

The reference CSV must contain `Algorithm`, `Problem`, and `MeanIGD` columns. Classification uses
the configured minimum coverage (default 80%), absolute difference $I_E-I_P$ tolerance (0.10), and
relative increase $(I_E-I_P)/I_P$ tolerance (2.0): `Improved`, `Preserved`, `IGD Degradation`, or
`Insufficient Coverage`.

`_fidelity_trial.py` is the internal one-seed worker used by this command.

## Computational scaling and speedup

`run_computational_scalability_benchmark.py` measures generated algorithms on DTLZ3. The default
execution mode is `torch.compile`. Use `--executions eager` when a generated algorithm cannot be
compiled. Compilation and warm-up are excluded from the timed region.

Every trial runs in a fresh subprocess. Execution is deliberately serial on one GPU to prevent CUDA
memory accumulation and `torch.compile` cache state from crossing algorithm or scale boundaries.

Run population scaling:

```bash
python evaluation/run_computational_scalability_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --scaling population \
  --gpu 0
```

Run dimension scaling:

```bash
python evaluation/run_computational_scalability_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --scaling dimension \
  --gpu 0
```

The defaults reproduce the scaling protocol:

| Setting | Population scaling | Dimension scaling |
|---|---:|---:|
| Problem | DTLZ3, 3 objectives | DTLZ3, 3 objectives |
| Fixed value | dimension = 100 | population = 1000 |
| Scales | 256 to 16384 | 1024 to 65536 |
| Timed generations | 100 | 100 |
| Warm-up generations | 5 | 5 |
| Repeats | 11 | 11 |

### Optional PlatEMO scaling baseline

Two MATLAB entry points run the corresponding PlatEMO baselines with the same 48 algorithm names,
DTLZ3 settings, scale values, 100 generations, and 11 repeats:

```bash
PLATEMO_ROOT=/path/to/PlatEMO matlab -batch \
  "run('evaluation/run_platemo_computational_scalability_population_benchmark.m')"

PLATEMO_ROOT=/path/to/PlatEMO matlab -batch \
  "run('evaluation/run_platemo_computational_scalability_dimension_benchmark.m')"
```

The results are written to `evaluation_results/platemo_scaling/`. Both scripts support checkpoint
resume, a three-hour timeout per repeat, and skipping larger scales after a timeout. Before a full
run, set `EVOCOCO_MATLAB_SMOKE_TEST=1` to test AGE-MOEA on a small DTLZ3 instance.

After generating the matching PlatEMO baseline, pass it to the EvoX command to calculate the
paper's speedup, $S=(R_P/G_P)/(R_E/G_E)$. For example:

```bash
python evaluation/run_computational_scalability_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --scaling population \
  --platemo-csv evaluation_results/platemo_scaling/platemo_population_scaling.csv \
  --gpu 0
```

Only repeat indices completed by both systems contribute to `speedup_x`. PlatEMO timeouts are
excluded from that value and reported separately as `timeout_lower_bound_speedup_x`. EvoX compile
results are preferred; eager results are used only as a fallback when no compiled trial succeeded
for a configuration. `compile_vs_eager_speedup_x`, when both modes were requested, is a separate
execution-mode diagnostic and is not the paper's PlatEMO/EvoX speedup.

Large settings can take hours or exceed GPU memory. Use this smoke test first:

```bash
python evaluation/run_computational_scalability_benchmark.py \
  --algorithm-file path/to/generated_algorithm.py \
  --scaling population \
  --sizes 100 \
  --repeats 1 \
  --warmup 1 \
  --generations 3 \
  --gpu 0 \
  --output-dir evaluation_results/scaling_smoke
```

Useful options:

- `--algorithms AGE-MOEA PREA` selects files by filename stem.
- `--executions compile` is the default; `eager` is the fallback and `both` additionally measures
  compile versus eager execution.
- `--timeout 3600` sets the per-trial timeout in seconds.
- `--device cpu` enables a functional CPU smoke test; reported GPU speedups require CUDA.

The output directory contains individual `trials.csv` records and a `summary.csv` with per-mode
means and, when `--platemo-csv` is supplied, the PlatEMO/EvoX paired speedup. IGD remains in the
output as an optimization-health diagnostic; this script does not turn eager/compile per-seed IGD
differences into a fidelity verdict.

`_scaling_trial.py` is the internal isolated worker used by this command.
