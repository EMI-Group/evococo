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

<p align="center">
  Automatically restructure evolutionary multiobjective optimization software into tensorized
  PyTorch/EvoX programs while preserving its underlying algorithmic semantics.
</p>

<div align="center">
  <a href="https://arxiv.org/abs/2503.20286">
    <img src="https://img.shields.io/badge/EvoMO%20paper-arXiv-red?style=for-the-badge" alt="EvoMO Paper on arXiv">
  </a>
  <a href="https://github.com/EMI-Group/evox">
    <img src="https://img.shields.io/badge/Built%20with-EvoX-C8383C?style=for-the-badge" alt="Built with EvoX">
  </a>
  <a href="https://github.com/EMI-Group/evococo/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/EMI-Group/evococo?style=for-the-badge" alt="License">
  </a>
</div>

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Quick Start](#quick-start)
6. [Experiments and Benchmarking](#experiments-and-benchmarking)
7. [Generated Algorithms](#generated-algorithms)
8. [Project Structure](#project-structure)
9. [Security Notice](#security-notice)
10. [Community and Support](#community-and-support)
11. [License](#license)

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
repository also contains 48 generated algorithm implementations used in the accompanying
experiments.

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
python evaluation/benchmark.py --help

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

Select one provider and add the corresponding API key:

```env
ACTIVE_LLM_PROVIDER=zhipu
OPENAI_TEMPERATURE=0.2

ZHIPU_API_KEY=your_zhipu_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
CUSTOM_API_KEY=your_custom_api_key_here
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

### Benchmark generated Python algorithms

Run the standard syntax, static, execution, and IGD checks on a directory of Python files:

```bash
python evaluation/benchmark.py \
  --dir experiments/generated_algorithms \
  --workers 1
```

The command writes `benchmark_report.json` into the evaluated directory. Increase `--workers`
carefully because each worker may allocate GPU memory.

### Run one DTLZ evaluation

```bash
python evaluation/run_dtlz_benchmark.py \
  --algo_file experiments/generated_algorithms/AGE-MOEA.py \
  --problem DTLZ2 \
  --seed 1 \
  --output results/age_moea_dtlz2.csv
```

Supported problems are DTLZ1 through DTLZ7.

## Generated Algorithms

The [`experiments/generated_algorithms`](experiments/generated_algorithms) directory contains the
48 EvoCoCo-generated Python implementations used as public experiment artifacts. They cover
decomposition-based, dominance-based, indicator-based, sparse, particle-swarm, and many-objective
optimization methods.

These files are preserved as generated outputs. Consequently, a small number retain Ruff warnings
such as unused imports or variables even though all 48 implementations pass syntax and execution
checks in the provided benchmark environment.

## Project Structure

```text
evococo/
├── backend/                         # FastAPI service and seven-stage tournament engine
│   ├── database/                    # Retrieval rules for common translation failures
│   └── prompts/                     # Agent roles, specifications, and EvoX resources
├── frontend/                        # Browser-based interactive interface
├── experiments/                     # Batch runners, baselines, and public artifacts
│   └── generated_algorithms/        # 48 generated EvoX implementations
├── evaluation/                      # Benchmark and DTLZ evaluation scripts
├── docs/images/                     # EvoX brand assets used by this README
├── requirements.txt
└── README.md
```

## Security Notice

> [!WARNING]
> EvoCoCo executes LLM-generated Python code during validation. Run it only in a trusted, isolated
> environment, review generated code before reuse, and do not expose the backend directly to an
> untrusted network or untrusted users.

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
