import os
import sys
import argparse
import asyncio
import re
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Automatically locate the .env file in the project root directory
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path)

# Import backend configuration dynamically
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import LLM_PROVIDERS

ACTIVE_PROVIDER = os.getenv("ACTIVE_LLM_PROVIDER", "custom")
if ACTIVE_PROVIDER in LLM_PROVIDERS:
    provider_config = LLM_PROVIDERS[ACTIVE_PROVIDER]
    API_KEY = os.getenv(provider_config["api_key_env"])
    BASE_URL = provider_config["base_url"]
    MODEL_NAME = provider_config["model"]
else:
    ACTIVE_PROVIDER = "custom"
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("OPENAI_BASE_URL")
    MODEL_NAME = os.getenv("OPENAI_MODEL")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT_ONESHOT = """You translate MATLAB multi-objective evolutionary algorithms into high-performance Python code using the EvoX framework format and PyTorch.

You must follow the EvoCoCo System Architecture Standard:
{GLOBAL_SPEC}

Here is the Migration Reference Guide containing shape/role forensic mappings:
{MIGRATION_REFERENCE}

Here's an example to show you the syntax of translating MATLAB evolutionary algorithms to EvoX:

The example MATLAB algorithm is:
```matlab
{NSGA2_MATLAB}
```

The example new translated PyTorch/EvoX code looks like this:
```python
{NSGA2_PYTHON}
```
"""

NSGA2_MATLAB_TEXT = r"""classdef NSGAII < ALGORITHM
    methods
        function main(Algorithm,Problem)
            Population = Problem.Initialization();
            [~,FrontNo,CrowdDis] = EnvironmentalSelection(Population,Problem.N);

            while Algorithm.NotTerminated(Population)
                MatingPool = TournamentSelection(2,Problem.N,FrontNo,-CrowdDis);
                Offspring  = OperatorGA(Problem,Population(MatingPool));
                [Population,FrontNo,CrowdDis] = EnvironmentalSelection([Population,Offspring],Problem.N);
            end
        end
    end
end

function [Population,FrontNo,CrowdDis] = EnvironmentalSelection(Population,N)
    [FrontNo,MaxFNo] = NDSort(Population.objs,Population.cons,N);
    Next = FrontNo < MaxFNo;

    CrowdDis = CrowdingDistance(Population.objs,FrontNo);

    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:N-sum(Next)))) = true;

    Population = Population(Next);
    FrontNo    = FrontNo(Next);
    CrowdDis   = CrowdDis(Next);
end
"""

NSGA2_PYTHON_TEXT = r"""from typing import Callable, Optional
import torch
from evox.core import Algorithm, Mutable, Parameter
from evox.utils import clamp, randint, nanmin, nanmax, lexsort
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit, crowding_distance
from evomo.operators.selection import nd_environmental_selection, non_dominate_rank, ref_vec_guided
from evomo.utils import unique_rows_sorted

class NSGA2(Algorithm):
    def __init__(
        self,
        pop_size: int,
        n_objs: int,
        lb: torch.Tensor,
        ub: torch.Tensor,
        selection_op: Optional[Callable] = None,
        mutation_op: Optional[Callable] = None,
        crossover_op: Optional[Callable] = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        device = torch.get_default_device() if device is None else device
        
        # Dimensions and Bounds
        self.dim = lb.shape[0]
        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)

        # Operators
        self.selection = selection_op if selection_op is not None else tournament_selection_multifit
        self.mutation = mutation_op if mutation_op is not None else polynomial_mutation
        self.crossover = crossover_op if crossover_op is not None else simulated_binary

        # Initialize State (Mutable)
        length = ub - lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.rank = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.dis = Mutable(torch.empty(self.pop_size, device=device).fill_(-torch.inf))

    def init_step(self):
        self.fit = self.evaluate(self.pop)
        self.pop, self.fit, self.rank, self.dis = self._environmental_selection(self.pop, self.fit)

    def _environmental_selection(self, pop, fit):
        # 1. Non-dominated Sorting (Constraint-handling stripped)
        rank = non_dominate_rank(fit)
        
        # 2. Vectorized Crowding Distance Front-by-Front (Standard JIT-compliant pattern)
        N_ext = fit.shape[0]
        cd = torch.zeros(N_ext, device=fit.device)
        for i in range(min(N_ext, self.pop_size + 1)):
            mask = (rank == i)
            if mask.any():
                # crowding_distance requires (fit, mask) parameters
                cd_front = crowding_distance(fit, mask)
                cd = torch.where(mask, cd_front, cd)

        # 3. Sort by Rank (ascending) and Crowding Distance (descending)
        sort_keys = torch.stack([-cd, rank.float()])
        indices = lexsort(sort_keys)[:self.pop_size]

        return pop[indices], fit[indices], rank[indices], cd[indices]

    def step(self):
        # 1. Selection & Mating
        mating_pool = self.selection(self.pop_size, [-self.dis, self.rank])
        crossovered = self.crossover(self.pop[mating_pool])
        
        # 2. Mutation & Repair
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)
        
        # 3. Evaluation
        off_fit = self.evaluate(offspring)
        
        # 4. Merge & Environmental Selection
        merge_pop = torch.cat([self.pop, offspring], dim=0)
        merge_fit = torch.cat([self.fit, off_fit], dim=0)

        self.pop, self.fit, self.rank, self.dis = self._environmental_selection(merge_pop, merge_fit)

# === FIXED DEMO BLOCK ===
if __name__ == "__main__":
    import time
    import torch
    from evox.metrics import igd
    from evox.problems.numerical import DTLZ2
    from evox.workflows import StdWorkflow

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    algo = NSGA2(pop_size=100, n_objs=3, lb=-torch.zeros(12), ub=torch.ones(12))
    prob = DTLZ2(m=3)
    pf = prob.pf()
    workflow = StdWorkflow(algo, prob)
    workflow.init_step()
    jit_state_step = torch.compile(workflow.step)

    # 1. Trigger JIT compilation (First step)
    jit_state_step()

    # 2. Pure execution (Remaining 49 steps)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exec_start = time.perf_counter()

    for i in range(1, 50):
        jit_state_step()

        if (i + 1) % 5 == 0:
            fit = workflow.algorithm.fit
            fit = fit[~torch.any(torch.isnan(fit), dim=1)]
            print(f"Gen {i + 1} IGD: {igd(fit, pf)}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    exec_time = time.perf_counter() - exec_start
    print(f"Execution time for Gen 2-50 (49 steps): {exec_time:.4f}s (Avg: {exec_time / 49:.4f}s/gen)")
"""

async def one_shot_translate(matlab_code: str) -> str:
    try:
        is_platemo = bool(re.search(r"classdef\s+[A-Za-z0-9_]+\s*<\s*ALGORITHM", matlab_code))
        ref_name = "reference_platemo.md" if is_platemo else "reference_custom.md"
        
        prompts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "prompts")
        
        with open(os.path.join(prompts_dir, "0_global_spec.md"), "r", encoding="utf-8") as f:
            global_spec_text = f.read()
            
        with open(os.path.join(prompts_dir, ref_name), "r", encoding="utf-8") as f:
            migration_ref_text = f.read()

        prompt = SYSTEM_PROMPT_ONESHOT.format(
            GLOBAL_SPEC=global_spec_text,
            MIGRATION_REFERENCE=migration_ref_text,
            NSGA2_MATLAB=NSGA2_MATLAB_TEXT,
            NSGA2_PYTHON=NSGA2_PYTHON_TEXT
        )

        kwargs_api = {}
        if ACTIVE_PROVIDER == "litellm":
            kwargs_api["extra_body"] = {"reasoning_effort": "minimal"}

        import time
        start_time = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user", 
                    "content": f"{prompt}\n\nYou are given the following MATLAB algorithm:\n\n{matlab_code}\n\nTranslate the algorithm into PyTorch/EvoX! Name your optimized class <YourAlgoName> (replace in class definition and the verification block). Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is functional. Just output the new model code, including the imports, class, helper functions, and the FIXED DEMO BLOCK at the end!"
                }
            ],
            temperature=0.0,
            timeout=400.0,
            stream=False,
            **kwargs_api,
        )
        latency = time.perf_counter() - start_time
        
        content = response.choices[0].message.content
        
        # Clean up any accidentally generated markdown code block tags
        content = re.sub(r"^```[a-zA-Z]*\s*\n", "", content)
        content = re.sub(r"\n```\s*$", "", content)
        content = content.replace("```", "").strip()

        usage = getattr(response, "usage", None)
        metrics = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
            "latency": latency,
        }
        return content, metrics
    except Exception as e:
        print(f"Translation Error: {e}")
        return "", {}

async def main():
    parser = argparse.ArgumentParser(description="One-shot MATLAB to Python Translation Baseline (Custom-aware)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input MATLAB file")
    parser.add_argument("--output", "-o", type=str, default="experiments/baselines/one_shot_output.py", help="Path to save output Python file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    matlab_code = ""
    if os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in sorted(files):
                if file.endswith('.m') or file.endswith('.txt'):
                    fpath = os.path.join(root, file)
                    with open(fpath, "r", encoding="utf-8") as f:
                        matlab_code += f"\n\n--- {file} ---\n{f.read()}"
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            matlab_code = f.read()

    print("=======================================")
    print(" [Baseline: One-Shot LLM Translation] ")
    print(f" Model: {MODEL_NAME}")
    print(f" Input: {args.input}")
    print("=======================================")

    python_code, metrics = await one_shot_translate(matlab_code)

    if python_code:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        stats_header = (
            "# ==============================================================================\n"
            "# One-Shot LLM Translation Statistics\n"
            f"# Latency: {metrics.get('latency', 0.0):.2f}s\n"
            f"# Prompt Tokens: {metrics.get('prompt_tokens', 0):,}\n"
            f"# Completion Tokens: {metrics.get('completion_tokens', 0):,}\n"
            f"# Total Tokens: {metrics.get('total_tokens', 0):,}\n"
            "# ==============================================================================\n\n"
        )
        python_code = stats_header + python_code
            
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(python_code)
        
        print(f"[Success] Translation saved to {args.output}")
    else:
        print("[Failed] Did not get code from LLM.")

if __name__ == "__main__":
    asyncio.run(main())
