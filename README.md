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
8. [Community and Support](#community-and-support)
9. [License](#license)

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
- Includes 48 MOEAs for evaluating migration reliability, optimization fidelity, computational
  scalability, external transfer, and component contributions.

## Installation

Clone the repository and install its Python dependencies:

```bash
git clone https://github.com/EMI-Group/evococo.git
cd evococo
python -m pip install -r requirements.txt
```

### Coding-agent installation

Give the following instruction to a coding agent with terminal access:

```text
Install and validate this EvoCoCo repository:
python -m pip install -r requirements.txt
python -m compileall -q backend evaluation experiments
python evaluation/run_migration_reliability_benchmark.py --help

Report the Python, PyTorch, EvoX, and EvoMO versions, CUDA availability, and validation results.
Do not call LLM APIs or run translation experiments during installation.
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

### Start the backend

From the project root, run:

```bash
python -m uvicorn backend.main:app --reload --reload-dir backend --port 8000
```

The backend health endpoint will be available at <http://localhost:8000>.

### Open the frontend

Open [`frontend/index.html`](frontend/index.html) in a browser. The frontend connects to the local
backend over WebSocket, displays every pipeline stage, and returns the selected Python
implementation when the tournament finishes.

> [!TIP]
> Keep the backend terminal open while using the browser interface so progress and error messages
> remain visible.

> [!NOTE]
> Validation executes generated Python code; use trusted inputs and run EvoCoCo locally or in an
> isolated environment.

## Experiments and Benchmarking

### Batch translation

Create an input directory containing `.m` or `.txt` files. A subdirectory containing multiple
MATLAB source files is treated as one algorithm:

```bash
mkdir -p experiments/matlab_code
python experiments/batch_translate.py \
  --input_dir experiments/matlab_code \
  --output_dir experiments/benchmark_results \
  --repeats 1 \
  --repeat-concurrency 1
```

Each pipeline run starts six candidate branches and can consume multiple LLM requests. Begin with
one repeat and one concurrent run when validating a new provider configuration.

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
│   ├── database/                    # Retrieval rules for common translation failures
│   └── prompts/                     # Agent roles, specifications, and EvoX resources
├── frontend/                        # Browser-based interactive interface
├── experiments/                     # Batch runners, baselines, and public artifacts
│   └── generated_algorithms/        # 48 generated tensorized EvoX algorithms
├── evaluation/                      # Reliability, fidelity, and scaling benchmarks
├── docs/images/                     # EvoX brand assets used by this README
├── requirements.txt
└── README.md
```

<!--
## Citing EvoCoCo

If you use EvoCoCo in your research, please cite the following paper. Add the publication metadata
when it is finalized.

```bibtex
@article{evococo,
  title   = {Semantics-Guided Automatic Tensorization for Evolutionary Multiobjective Optimization: A Multi-Agent Framework},
  author  = {Liang, Zhenyu and Huang, Beichen and Zheng, Bowen and Cheng, Ran},
  year    = {2026}
}
```
-->

## Community and Support

Questions, bug reports, and feature requests are welcome through
[GitHub Issues](https://github.com/EMI-Group/evococo/issues). For EvoX framework questions, see the
[EvoX repository](https://github.com/EMI-Group/evox).

## License

EvoCoCo is released under the [GNU General Public License v3.0](LICENSE).
