import os
import argparse
import asyncio
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Automatically locate the .env file in the project root directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT_WEAK = """You are an expert algorithm translation assistant. Your task is to translate the provided MATLAB code into high-performance Python code using numpy/pytorch where appropriate.
Return ONLY valid Python code. Do not provide explanations, no markdown tags like ```python, just the raw code.
"""

SYSTEM_PROMPT_STRONG = """You are an expert algorithm translation assistant. 
Your MUST translate the provided MATLAB code into high-performance Python code using the EvoX framework format and PyTorch/JAX.
Return ONLY valid Python code. Do not provide explanations, no markdown tags like ```python, just the raw code.

# EVOX FRAMEWORK SPECIFICATION:
{GLOBAL_SPEC}
"""

GLOBAL_SPEC_TEXT = r"""# System Architecture Standard (EvoCoder)

## 0. Target Environment (Unconstrained Only)
* **Optimization Type**: Unconstrained Evolutionary Optimization.
* **Constraint Handling**: **STRICTLY FORBIDDEN**.
    * **Remove Logic**: Any logic related to constraint violation (`CV`), feasibility matrices, or penalty functions in the source MATLAB code **MUST BE REMOVED**.
    * **Assume Feasible**: The converted Python code must treat the problem as having **no constraints** (i.e., every solution is feasible).
    * **Objective Sorting**: Sorting and selection must rely **solely on Objective Values** (Fitness).

## 1. Input Interface
The algorithm is a class inheriting from `evox.Algorithm`.
It must accept a `problem` object and configuration parameters.

## 2. Tensor Standards (PyTorch)
All calculations must use `torch.Tensor` on GPU (cuda) if available.
* **Variable Naming (STRICT)**:
    * **Population (Decision Vars)**: MUST be named `self.pop`. Shape `(N, D)`.
    * **Fitness (Objective Vals)**: MUST be named `self.fit`. Shape `(N, M)`.
* **Bounds**: `lower_bound` and `upper_bound` are 1D tensors of shape `(D,)`.

## 3. Key Methods
* `init_step(self)`: 
    * Initialize internal states.
    * **Must set**: `self.pop` and `self.fit`.
* `step(self)`: 
    * Perform one iteration (mating -> mutation -> selection).
    * Update `self.pop` and `self.fit` **in-place**.

## 4. Forbidden Anti-Patterns (STRICTLY PROHIBITED)

1. **NO Individual-level Loops**: Do NOT write `for i in range(N):` to process individuals (e.g., mutation, crossover).
    * **❌ BAD**: `for i in range(N): if rand() < prob: mask[i] = toggle()`
    * **✅ GOOD (God-Mode)**: `do_mut = torch.rand(N) < prob; mask[do_mut] = toggle()`
2. **NO CPU-GPU Sync**: Do NOT use `.item()` or `.tolist()` in the main execution paths. This forces the GPU to halt and wait for the CPU.
3. **NO Dynamic Lists Extraction**: Do NOT use `selected_indices = []` and `extend()`. Operations must stay in Tensor space.
4. **NO Iterative Selection (While loop peeling)**: Do NOT write `while len(selected) < N:` to iteratively pick fronts.
    * **✅ GOOD (Lexsort)**: Calculate all metrics (e.g., Rank, Crowding Distance) simultaneously for **all** individuals. Then use `evox.utils.lexsort` or compound scoring (`Rank - CD * 1e-4`) with `torch.argsort` to rank all individuals in one single operation. Slice the top `self.pop = pop[sorted_idx[:N]]`.
5. **NO Numpy**: Use `torch` exclusively unless absolutely necessary.

## 5. Required Tensorization Paradigms
* **Random Sampling**: Instead of looping to find targets, use `torch.where`, boolean masking (`x[mask] = ...`), or `torch.multinomial` for batched probabilistic sampling without `for` loops.
* **In-Place Updates**: Instead of `X = X + Y`, use `X.add_(Y)` or `X += Y` if memory profiling is tight.
* **Component Re-use**: Do not manually rewrite standard functions if EvoX/EvoMo (`non_dominate_rank`, `crowding_distance`) can process the variables directly in a batched manner.

## 6. Algorithmic Fidelity (ABSOLUTE RULE)
* **NO "SIMPLIFIED FOR VECTORIZATION"**: You MUST 100% faithfully translate the mathematical objectives, fitness calculations, and archive pruning from the original MATLAB code. 
* **EXACT MATHEMATICAL ISOMORPHISM**: When translating scientific and mathematical elements (e.g., Jacobian matrices, multi-term fitness weighting, penalty formulas), you must perform a strict operator-by-operator mapping. NEVER alter the original code's constants (e.g., if MATLAB has `theta=0.2`, do not invent `theta=0.5` or hardcode your own assumptions). NEVER reduce multi-term formulas (e.g., if MATLAB uses `dis(:,1) + 0.1*dis(:,2)`, use `topk(k=2)` or `sort` to perfectly replicate the first and second neighbor weights; do NOT substitute it with a naive `min()` operation).
* **Do NOT** simplify complex metrics (e.g., higher-order distances, multi-stage selections, dual-fitness contributions) into single-dimension proximities just because it is easier to write in PyTorch without a loop. 
* **NO PYTHON FOR-LOOP FALLBACKS**: If tensorizing a highly complex evaluation formula seems mathematically difficult, you **MUST NEVER** fall back to a lightweight Python `for` loop over individuals (N) or dimensions (D). You MUST force vectorization using advanced boolean indexing (`mask = ...; tensor[mask] = ...`), broadcasting (`A.unsqueeze(1) - B.unsqueeze(0)`), or `torch.where`. You must execute the original mathematics entirely via batched tensor operations without sacrificing ecological logic.
"""

async def zero_shot_translate(matlab_code: str, mode: str) -> str:
    try:
        if mode == 'strong':
            prompt = SYSTEM_PROMPT_STRONG.replace("{GLOBAL_SPEC}", GLOBAL_SPEC_TEXT)
        else:
            prompt = SYSTEM_PROMPT_WEAK

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nTranslate this MATLAB code to Python:\n\n{matlab_code}"
                }
            ],
            temperature=0.0,
            timeout=60.0,
            extra_body={"reasoning_effort": "minimal"},
        )
        content = response.choices[0].message.content
        
        # Clean up any accidentally generated markdown code block tags
        content = re.sub(r"^```[a-zA-Z]*\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.replace("```", "").strip()
        return content
    except Exception as e:
        print(f"Translation Error: {e}")
        return ""

async def main():
    parser = argparse.ArgumentParser(description="Zero-shot MATLAB to Python Translation Baseline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MATLAB file")
    parser.add_argument("--output", "-o", type=str, default="experiments/baselines/zero_shot_output.py", help="Path to save output Python file")
    parser.add_argument("--mode", type=str, choices=["weak", "strong"], default="strong", help="Baseline mode: 'weak' or 'strong'")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    with open(args.input, "r", encoding="utf-8") as f:
        matlab_code = f.read()

    print("=======================================")
    print(" [Baseline: Zero-Shot LLM Translation] ")
    print(f" Mode:  {args.mode.upper()} Baseline")
    print(f" Model: {MODEL_NAME}")
    print(f" Input: {args.input}")
    print("=======================================")

    python_code = await zero_shot_translate(matlab_code, args.mode)

    if python_code:
        # Automatically create directories if the output path contains folders
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(python_code)
        
        print(f"[Success] Translation saved to {args.output}")
    else:
        print("[Failed] Did not get code from LLM.")

if __name__ == "__main__":
    asyncio.run(main())
