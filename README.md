<h1 align="center">
  <a href="https://github.com/EMI-Group/evox">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/images/evox_brand_light.svg">
      <source media="(prefers-color-scheme: light)" srcset="docs/images/evox_brand_dark.svg">
      <img alt="EvoX Logo" height="128" width="500" src="docs/images/evox_brand_dark.svg">
    </picture>
  </a>
</h1>

<h2 align="center">
  🌟 EvoCoCo: A Multi-Agent Framework for Semantics-Guided Automatic Tensorization 🌟
</h2>

<div align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/EvoCoCo%20paper-arXiv-red?style=for-the-badge" alt="EvoCoCo Paper on arXiv">
  </a>
</div>

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick Start](#quick-start)
6. [Experiments and Benchmarking](#experiments-and-benchmarking)
7. [Project Structure](#project-structure)
8. [Citing EvoCoCo](#citing-evococo)
9. [Community and Support](#community-and-support)
10. [License](#license)

## Overview

EvoCoCo is a multi-agent framework for automatically tensorizing evolutionary multiobjective
optimization (EMO) software. Rather than treating tensorization as syntax-to-syntax translation,
EvoCoCo formulates it as **semantics-guided computational restructuring**: the concrete MATLAB
implementation is the transformation object, while the underlying optimization mechanism is the
semantic constraint that the generated program must preserve.

Built on [EvoX](https://github.com/EMI-Group/evox), EvoCoCo coordinates semantic analysis,
contextual rule retrieval, transformation planning, diversified tensor restructuring, static and
runtime repair, and candidate selection through shared intermediate representations and
closed-loop execution feedback.

The system can be used through a browser interface or as a batch experiment runner. This
repository includes 48 EvoCoCo-generated tensorized algorithms used in the accompanying
experiments. The same algorithms are also integrated into
[EvoMO](https://github.com/EMI-Group/evomo).

## Key Features

### 🧭 Semantics-Guided Restructuring

- Treats the source implementation as the transformation object and the underlying optimization
  mechanism as the semantic constraint.
- Reconstructs algorithm states, operators, dependencies, and control flow before tensorization.

### 🤝 Multi-Agent Tensorization

- Coordinates semantic analysis, contextual rule retrieval, transformation planning,
  restructuring, repair, and selection through shared intermediate representations.
- Explores six blueprint-guided computational restructuring strategies under the same semantic
  constraints.

### ⚡ High-Performance Generated Algorithms

- Produces GPU-oriented PyTorch/EvoX implementations using broadcasting, `einsum`, masked
  operations, advanced tensor operators, and JIT-oriented restructuring.
- Achieves speedups of up to **10,000×** for generated tensorized algorithms in the reported
  experiments.

### 🧪 Closed-Loop Validation and Benchmarking

- Combines Ruff checks, runtime execution, optimization feedback, repair, and candidate selection.
- Includes 48 MOEAs for evaluating migration reliability, optimization fidelity, and
  computational scalability.

## Installation

Clone the repository and install its Python dependencies:

```bash
git clone https://github.com/EMI-Group/evococo.git
cd evococo
python -m pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for generated-algorithm evaluation. The single-run DTLZ
evaluator can also select CPU automatically, although GPU execution is the primary target of the
tensorized implementations.

## Configuration

Copy the example configuration before starting EvoCoCo:

```bash
cp .env.example .env
```

The following example uses Gemini:

```env
ACTIVE_LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
```

Available provider names are `zhipu`, `deepseek-v4-pro`, `deepseek-v4-flash`, `gemini`, and
`custom`. Optional base URL and model overrides are documented in
[`.env.example`](.env.example).

> [!IMPORTANT]
> Never commit `.env` or expose provider API keys in generated artifacts and logs.

## Quick Start

EvoCoCo can generate algorithms through either the browser interface or the command line.

### Method 1: Web interface

Start the backend from the project root:

```bash
python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

The backend health endpoint will be available at <http://localhost:8000>.

Open [`frontend/index.html`](frontend/index.html) in a browser. The frontend connects to the local
backend over WebSocket. Paste the MATLAB source code into the input panel and click **Run**. The
interface displays every pipeline stage and returns the selected Python implementation when the
tournament finishes.

> [!TIP]
> Keep the backend terminal open while using the browser interface so progress and error messages
> remain visible.

### Method 2: Command line

The command-line workflow does not require the Web backend. Create `experiments/single_input/` and
place one MATLAB `.m` file (or one directory of related `.m`/`.txt` files) inside it. Then run the
full EvoCoCo pipeline once:

```bash
python experiments/batch_translate.py \
  --input_dir experiments/single_input \
  --output_dir experiments/single_output \
  --repeats 1 \
  --repeat-concurrency 1
```

The generated algorithm is written to `experiments/single_output/<algorithm>_run1.py`, with run
statistics in the adjacent `<algorithm>_run1_stats.json`. Detailed intermediate artifacts are
retained under `run_history/`.

## Experiments and Benchmarking

### Batch translation

The command-line workflow above also supports batch generation. Place multiple `.m`/`.txt` files
in the input directory; each file is treated as one algorithm. A subdirectory containing related
source files is also treated as one algorithm. Use `--repeats` for repeated generations and
`--repeat-concurrency` to control how many repeats run concurrently.

### Evaluation benchmarks

Validate syntax, execution, and convergence of the 48 selected implementations:

```bash
python evaluation/run_migration_reliability_benchmark.py \
  --dir experiments/generated_algorithms \
  --workers 1
```

Run the optimization-fidelity benchmark on a selected suite:

```bash
python evaluation/run_optimization_fidelity_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --suite DTLZ \
  --runs 21 \
  --gpu 0
```

Run computational scaling with `torch.compile`:

```bash
python evaluation/run_computational_scalability_benchmark.py \
  --algorithm-dir experiments/generated_algorithms \
  --scaling population \
  --gpu 0
```

All benchmark runs are resumable. PlatEMO reference generation, the DTLZ/WFG/LSMOP/MaF options,
dimension scaling, speedup calculation, output fields, and smoke tests are documented in
[`evaluation/README.md`](evaluation/README.md).

## Project Structure

```text
evococo/
├── backend/                         # FastAPI service and seven-stage tournament engine
│   ├── engine.py                    # Pipeline orchestration (single-branch lifecycle, judge)
│   ├── executor.py                  # Static (Ruff) + runtime execution/repair
│   ├── generator.py                 # LLM provider adapters (zhipu/deepseek/gemini/custom)
│   ├── storage.py                   # RAG DB / spec / resource loading, artifact I/O
│   ├── stats.py                     # Performance metrics aggregation + reports
│   ├── config.py                    # Environment-driven configuration (load-bearing names)
│   ├── main.py                      # FastAPI app: / and /ws endpoints
│   ├── database/                    # Retrieval rules for common translation failures
│   └── prompts/                     # Agent roles, specifications, and EvoX resources
├── frontend/                        # Self-contained browser interface (no CDN)
├── experiments/                     # Batch/one-shot runners, baselines, and public artifacts
│   ├── batch_translate.py           # Multi-algorithm pipeline runner
│   ├── batch_one_shot.py            # Batch one-shot translation
│   ├── one_shot.py                  # Single-shot translation
│   ├── _common.py                   # Shared helpers (env setup, source discovery, JSON I/O)
│   └── generated_algorithms/        # 48 generated tensorized EvoX algorithms (read-only)
├── evaluation/                      # Reliability, fidelity, and scaling benchmarks
│   ├── run_migration_reliability_benchmark.py
│   ├── run_optimization_fidelity_benchmark.py
│   ├── run_computational_scalability_benchmark.py
│   ├── _common.py                   # Shared CLI/discovery/CSV/subprocess helpers
│   └── _benchmark_problems.py, _*_trial.py   # Problem definitions and worker trials
├── docs/
│   └── images/                      # EvoX brand assets used by this README
├── requirements.txt
└── README.md
```

## Citing EvoCoCo

If you use EvoCoCo in your research, please cite the
[arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX):

```bibtex
@article{evococo,
  title         = {Semantics-Guided Automatic Tensorization for Evolutionary Multiobjective Optimization: A Multi-Agent Framework},
  author        = {Liang, Zhenyu and Huang, Beichen and Zheng, Bowen and Cheng, Ran},
  journal       = {arXiv preprint arXiv:XXXX.XXXXX},
  year          = {2026}
}
```

## Community and Support

Questions, bug reports, and feature requests are welcome through
[GitHub Issues](https://github.com/EMI-Group/evococo/issues). For EvoX framework questions, see the
[EvoX repository](https://github.com/EMI-Group/evox).

## License

EvoCoCo is released under the [GNU General Public License v3.0](LICENSE).
